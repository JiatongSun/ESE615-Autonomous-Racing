#include "scan_matching_skeleton/correspond.h"
#include "cmath"
#include <iostream>

using namespace std;

const int UP_SMALL = 0;
const int UP_BIG = 1;
const int DOWN_SMALL = 2;
const int DOWN_BIG = 3;

const float LASER_MIN = -2.35;
const float LASER_MAX = 2.35;
const int LASER_RAYS = 1080;
const float LASER_RANGE = LASER_MAX - LASER_MIN;

void getNaiveCorrespondence(vector<Point> &old_points, vector<Point> &trans_points, vector<Point> &points,
                            vector<vector<int>> &jump_table, vector<Correspondence> &c, float prob) {

    c.clear();
    int last_best = -1;
    const int n = (int) trans_points.size();
    const int m = (int) old_points.size();
    int min_index = 0;
    int second_min_index = 0;

    // Do for each point
    for (int ind_trans = 0; ind_trans < n; ++ind_trans) {
        float min_dist = 100000.00;
        for (int ind_old = 0; ind_old < m; ++ind_old) {
            float dist = old_points[ind_trans].distToPoint2(&trans_points[ind_old]);
            if (dist < min_dist) {
                min_dist = dist;
                min_index = ind_old;
                if (ind_old == 0) {
                    second_min_index = ind_old + 1;
                } else {
                    second_min_index = ind_old - 1;
                }
            }
        }
        c.emplace_back(&trans_points[ind_trans], &points[ind_trans],
                       &old_points[min_index], &old_points[second_min_index]);
    }
}

void getCorrespondence(vector<Point> &old_points, vector<Point> &trans_points, vector<Point> &points,
                       vector<vector<int>> &jump_table, vector<Correspondence> &c, float prob) {

    // Written with inspiration from: https://github.com/AndreaCensi/csm/blob/master/sm/csm/icp/icp_corr_tricks.c
    // use helper functions and structs in transform.h and correspond.h
    // input : old_points : vector of struct points containing the old points (points of the previous frame)
    // input : trans_points : vector of struct points containing the new points transformed to the previous frame using the current estimated transform
    // input : points : vector of struct points containing the new points
    // input : jump_table : jump table computed using the helper functions from the transformed and old points
    // input : c: vector of struct correspondences . This is a reference which needs to be updated in place and return the new correspondences to calculate the transforms.
    // output : c; update the correspondence vector in place which is provided as a reference. you need to find the index of the best and the second-best point.

    // Initialize correspondences
    c.clear();
    int last_best = -1;
    const int trans_size = (int) trans_points.size();
    const int old_size = (int) old_points.size();
    int nrays = min(old_size, trans_size);

    // Do for each point
    for (int ind_trans = 0; ind_trans < nrays; ++ind_trans) {
        /// TODO: Implement Fast Correspondence Search

        // Get transformed point p_{i}^{w}
        Point p_i_w = trans_points[ind_trans];

        // Current best match, and its distance
        int best = -1;
        double best_dist = INT_MAX;

        // Search domain for best
        int from = 0;
        int to = nrays - 1;

        // Approximated index in scan y_{t−1} corresponding to point p^{w}
        int start_index = (int) ((p_i_w.theta - LASER_MIN) * (LASER_RAYS / LASER_RANGE));

        // If last match was successful, then start at that index + 1
        int we_start_at = (last_best != -1) ? (last_best + 1) : start_index;
        if (we_start_at > to) we_start_at = to;
        if (we_start_at < from) we_start_at = from;

        // Search is conducted in two directions: up and down
        int up = we_start_at + 1, down = we_start_at;

        // Distance of last point examined in the up (down) direction
        double last_dist_up = 0, last_dist_down = -1;

        // True if search is finished in the up (down) direction
        bool up_stopped = false, down_stopped = false;

        // Until the search is stopped in both directions
        while ((!up_stopped) || (!down_stopped)) {
            // Should we try to explore up or down?
            bool now_up = !up_stopped && (down_stopped || last_dist_up < last_dist_down);

            // Now two symmetric chunks of code, the now_up and the !now_up
            if (now_up) {
                // If we have finished the points to search, we stop
                if (up > to) {
                    up_stopped = true;
                    continue;
                }

                // This is the distance from p_{i}^{W} to the up point
                last_dist_up = p_i_w.distToPoint2(&old_points[up]);

                // If it is less than the best point, up is our best guess so far
                if ((last_dist_up < best_dist) || (best == -1)) {
                    best = up;
                    best_dist = last_dist_up;
                }

                if (up > start_index) {
                    // If we are moving away from `start_index` we can compute a bound for early stopping.
                    // Currently, our best point has distance `best_dist`
                    // We can compute the minimum distance to p_{i}^{w} for points j > up
                    double delta_theta = (old_points[up].theta - p_i_w.theta);
                    double min_dist_up = p_i_w.r * ((delta_theta > M_PI_2) ? 1 : sin(delta_theta));

                    if (pow(min_dist_up, 2) > best_dist) {
                        // If going up we can’t make better than `best_dist`,
                        // then we stop searching in the "up" direction
                        up_stopped = true;
                        continue;
                    }

                    // If we are moving away, then we can implement the jump tables optimization
                    // If p_i_w is longer than "up", we can jump to a bigger/further point,
                    // or else we jump to a smaller/closer point
                    up = (old_points[up].r < p_i_w.r) ? jump_table[up][UP_BIG] : jump_table[up][UP_SMALL];
                } else {
                    // If we are moving towards "start_cell", we can’t do any of the
                    // previous optimizations, and we just move to the next point.
                    ++up;
                }
            }


            // This is the specular part of the previous chunk of code
            if (!now_up) {
                if (down < from) {
                    down_stopped = true;
                    continue;
                }
                last_dist_down = p_i_w.distToPoint2(&old_points[down]);
                if ((last_dist_down < best_dist) || (best == -1)) {
                    best = down;
                    best_dist = last_dist_down;
                }
                if (down < start_index) {
                    double delta_theta = (p_i_w.theta - old_points[down].theta);
                    double min_dist_down = p_i_w.r * ((delta_theta > M_PI_2) ? 1 : sin(delta_theta));

                    if (pow(min_dist_down, 2) > best_dist) {
                        down_stopped = true;
                        continue;
                    }

                    down = (old_points[down].r < p_i_w.r) ? jump_table[down][DOWN_BIG] : jump_table[down][DOWN_SMALL];
                } else {
                    --down;
                }
            }
        }

        // Now we want to find the second-best match
        // We find the next valid point, up and down
        // And then (very boring) we use the nearest
        int second_best;
        int second_best_up = best + 1;
        int second_best_down = best - 1;
        double dist_up = p_i_w.distToPoint2(&old_points[second_best_up]);
        double dist_down = p_i_w.distToPoint2(&old_points[second_best_down]);
        second_best = dist_up < dist_down ? second_best_up : second_best_down;

        // For the next point, we will start at best
        last_best = best;

        c.emplace_back(&trans_points[ind_trans], &points[ind_trans],
                       &old_points[best], &old_points[second_best]);

    }
}

void computeJump(vector<vector<int>> &table, vector<Point> &points) {
    table.clear();
    int n = (int) points.size();
    for (int i = 0; i < n; ++i) {
        vector<int> v = {n, n, -1, -1};
        for (int j = i + 1; j < n; ++j) {
            if (points[j].r < points[i].r) {
                v[UP_SMALL] = j;
                break;
            }
        }
        for (int j = i + 1; j < n; ++j) {
            if (points[j].r > points[i].r) {
                v[UP_BIG] = j;
                break;
            }
        }
        for (int j = i - 1; j >= 0; --j) {
            if (points[j].r < points[i].r) {
                v[DOWN_SMALL] = j;
                break;
            }
        }
        for (int j = i - 1; j >= 0; --j) {
            if (points[j].r > points[i].r) {
                v[DOWN_BIG] = j;
                break;
            }
        }
        table.push_back(v);
    }
}
