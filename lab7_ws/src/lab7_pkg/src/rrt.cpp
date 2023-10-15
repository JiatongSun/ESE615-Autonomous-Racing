#include "rrt.h"

RRT::RRT() : gen((random_device()) ()) {}

RRT::RRT(const vector<double> &init,
         const vector<double> &goal,
         const vector<vector<vector<double>>> &occupancy_grid,
         double collision_tol,
         double expand_dis,
         double path_resolution,
         double goal_sample_rate,
         int max_iter,
         double circle_dist,
         bool early_stop,
         bool smooth) :
        gen((random_device()) ()),
        collision_tol(collision_tol),
        expand_dis(expand_dis),
        path_resolution(path_resolution),
        goal_sample_rate(goal_sample_rate),
        max_iter(max_iter),
        circle_dist(circle_dist),
        early_stop(early_stop),
        smooth(smooth) {
    start = TreeNode(init[0], init[1]);
    start.is_root = true;
    end = TreeNode(goal[0], goal[1]);

    int nx = (int) occupancy_grid[0].size();
    int ny = (int) occupancy_grid[0][0].size();
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (occupancy_grid[0][i][j] == 0.0) continue;
            occupied_x.push_back(occupancy_grid[1][i][j]);
            occupied_y.push_back(occupancy_grid[2][i][j]);
            num_occupied++;
        }
    }

    double x1 = occupancy_grid[1][0][0];
    double x2 = occupancy_grid[1][nx - 1][ny - 1];
    xmin = x1 > x2 ? x2 : x1;
    xmax = x1 > x2 ? x1 : x2;

    double y1 = occupancy_grid[2][0][0];
    double y2 = occupancy_grid[2][nx - 1][ny - 1];
    ymin = y1 > y2 ? y2 : y1;
    ymax = y1 > y2 ? y1 : y2;

    x_dist = uniform_real_distribution<>(xmin, xmax);
    y_dist = uniform_real_distribution<>(ymin, ymax);
    percent = uniform_real_distribution<>(0.0, 100.0);
}

void RRT::planning() {
    // Check whether start point can reach goal point directly
    start.close_to_goal = is_goal(start, end);
    if (start.close_to_goal) {
        path = {start, end};
        smooth_path = {start, end};
        return;
    }

    tree.push_back(start);
    for (int i = 0; i < max_iter; ++i) {
        vector<double> rnd_point = sample();
        int nearest_idx = nearest(rnd_point);
        TreeNode nearest_node = tree[nearest_idx];

        TreeNode new_node = steer(nearest_node, rnd_point);
        new_node.parent = nearest_idx;
        new_node.cost = nearest_node.cost + line_cost(nearest_node, new_node);

        if (check_collision(nearest_node, new_node)) continue;

        // Connect along a minimum-cost path
//        vector<int> neighborhood = near(new_node);
//        int parent_idx = choose_parent(new_node, neighborhood);
//
//        if (parent_idx >= 0) {
//            TreeNode parent_node = tree[parent_idx];
//            new_node.parent = parent_idx;
//            new_node.cost = parent_node.cost + line_cost(parent_node, new_node);
//        }

        // Rewire the tree
//        rewire(new_node, neighborhood);
        tree.push_back(new_node);

        // Check goal reachability
        TreeNode *latest_node = &tree[tree.size() - 1];
        latest_node->close_to_goal = is_goal(*latest_node, end);
        if (early_stop && latest_node->close_to_goal) {
            path = find_path(*latest_node);
            if (smooth) {
                smooth_path = path_smoother(path);
            }
            return;
        }
    }

    int last_idx = search_best_goal_node();
    if (last_idx < 0) {
        return;
    }
    path = find_path(tree[last_idx]);
    if (smooth) {
        smooth_path = path_smoother(path);
    }
}

vector<double> RRT::sample() {
    double x, y;
    if (percent(gen) > goal_sample_rate) {
        x = x_dist(gen);
        y = y_dist(gen);
    } else {
        x = end.x;
        y = end.y;
    }

    return {x, y};
}

int RRT::nearest(const vector<double> &sampled_point) {
    int min_idx;
    double min_dist = 1e8;
    for (int i = 0; i < (int) tree.size(); ++i) {
        double dx = tree[i].x - sampled_point[0];
        double dy = tree[i].y - sampled_point[1];
        double dist = sqrt(dx * dx + dy * dy);
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = i;
        }
    }

    return min_idx;
}

TreeNode RRT::steer(const TreeNode &nearest_node, vector<double> sampled_point) const {
    double dx = sampled_point[0] - nearest_node.x;
    double dy = sampled_point[1] - nearest_node.y;
    double dist = sqrt(dx * dx + dy * dy);
    double theta = atan2(dy, dx);

    double x, y;
    if (dist > expand_dis) {
        x = nearest_node.x + expand_dis * cos(theta);
        y = nearest_node.y + expand_dis * sin(theta);
    } else {
        x = sampled_point[0];
        y = sampled_point[1];
    }

    TreeNode new_node(x, y);

    return new_node;
}

bool RRT::check_collision(const TreeNode &nearest_node, const TreeNode &new_node) {
    double dx = new_node.x - nearest_node.x;
    double dy = new_node.y - nearest_node.y;
    double dist = sqrt(dx * dx + dy * dy);
    double theta = atan2(dy, dx);

    int n_pts = int(dist / path_resolution) + 1;
    double resolution = dist / (n_pts - 1);

    for (int i = 0; i < n_pts; ++i) {
        double x = nearest_node.x + resolution * i * cos(theta);
        double y = nearest_node.y + resolution * i * sin(theta);
        if (dist_to_grid({x, y}) < collision_tol) {
            return true;
        }
    }

    return false;
}

bool RRT::is_goal(const TreeNode &latest_added_node, const TreeNode &goal_node) {
    double dx = latest_added_node.x - goal_node.x;
    double dy = latest_added_node.y - goal_node.y;
    double dist = sqrt(dx * dx + dy * dy);
    if (dist > expand_dis) {
        return false;
    }

    return !check_collision(latest_added_node, goal_node);
}

vector<TreeNode> RRT::find_path(const TreeNode &latest_added_node) {
    vector<TreeNode> raw_path;
    if (latest_added_node.x != end.x || latest_added_node.y != end.y) {
        raw_path.push_back(end);
    }

    TreeNode tmp_node = latest_added_node;
    while (!tmp_node.is_root) {
        raw_path.push_back(tmp_node);
        tmp_node = tree[tmp_node.parent];
    }

    raw_path.push_back(start);
    reverse(raw_path.begin(), raw_path.end());

    return raw_path;
}

double RRT::dist_to_grid(vector<double> point) {
    double min_dist = 1e8;
    for (int i = 0; i < num_occupied; ++i) {
        double dx = point[0] - occupied_x[i];
        double dy = point[1] - occupied_y[i];
        double dist = sqrt(dx * dx + dy * dy);
        if (dist < min_dist) {
            min_dist = dist;
        }
    }
    return min_dist;
}

int RRT::choose_parent(TreeNode &new_node, const vector<int> &neighborhood) {
    int min_idx;
    double min_cost = 1e8;
    for (const auto idx: neighborhood) {
        TreeNode neighbor_node = tree[idx];
        if (!check_collision(neighbor_node, new_node)) {
            double tmp_cost = cost(neighbor_node, new_node);
            if (tmp_cost < min_cost) {
                min_cost = tmp_cost;
                min_idx = idx;
            }
        }
    }

    if (min_cost == 1e8) {
        return -1;
    }

    return min_idx;
}

void RRT::rewire(const TreeNode &new_node, const vector<int> &neighborhood) {
    for (const auto idx: neighborhood) {
        TreeNode neighbor_node = tree[idx];
        if (check_collision(new_node, neighbor_node)) continue;

        double tmp_cost = cost(new_node, neighbor_node);
        if (neighbor_node.cost <= tmp_cost) continue;

        tree[idx].cost = tmp_cost;
        tree[idx].parent = (int) tree.size();
        propagate_cost_to_leaves(idx);
    }
}

int RRT::search_best_goal_node() {
    int min_idx = -1;
    double min_cost = 1e8;
    for (int i = 0; i < (int) tree.size(); ++i) {
        if (!tree[i].close_to_goal) continue;
        double tmp_cost = cost(tree[i], end);
        if (tmp_cost < min_cost) {
            min_cost = tmp_cost;
            min_idx = i;
        }
    }

    return min_idx;
}

double RRT::cost(const TreeNode &from_node, const TreeNode &to_node) {
    return from_node.cost + line_cost(from_node, to_node);
}

double RRT::line_cost(const TreeNode &n1, const TreeNode &n2) {
    double dx = n1.x - n2.x;
    double dy = n1.y - n2.y;
    double dist = sqrt(dx * dx + dy * dy);

    return dist;
}

void RRT::propagate_cost_to_leaves(int parent) {
    for (int i = 0; i < (int) tree.size(); ++i) {
        if (tree[i].parent == parent) {
            tree[i].cost = cost(tree[parent], tree[i]);
            propagate_cost_to_leaves(i);
        }
    }
}

vector<int> RRT::near(const TreeNode &node) {
    int n_node = (int) tree.size() + 1;
    double r = circle_dist * sqrt(log(n_node) / n_node);
    if (r > expand_dis) {
        r = expand_dis;
    }

    vector<int> neighborhood;
    for (int i = 0; i < (int) tree.size(); ++i) {
        double dx = tree[i].x - node.x;
        double dy = tree[i].y - node.y;
        double dist = sqrt(dx * dx + dy * dy);
        if (dist <= r) {
            neighborhood.push_back(i);
        }
    }

    return neighborhood;
}

vector<TreeNode> RRT::path_smoother(const vector<TreeNode> &raw_path) {
    int n = (int) raw_path.size();
    vector<double> dp(n, 1e8);

    vector<int> parents(n);
    for (int i = 0; i < n - 1; ++i) {
        parents[i + 1] = i;
    }

    for (int i = 0; i < n; ++i) {
        if (!check_collision(raw_path[0], raw_path[i])) {
            dp[i] = line_cost(raw_path[0], raw_path[i]);
            parents[i] = 0;
            continue;
        }

        for (int j = 1; j < i; ++j) {
            if (check_collision(raw_path[j], raw_path[i])) {
                continue;
            }
            double tmp_cost = dp[j] + line_cost(raw_path[j], raw_path[i]);
            if (tmp_cost < dp[i]) {
                dp[i] = tmp_cost;
                parents[i] = j;
            }
        }
    }

    vector<TreeNode> new_path;
    int node_idx = n - 1;
    while (node_idx != 0) {
        new_path.push_back(raw_path[node_idx]);
        node_idx = parents[node_idx];
    }
    new_path.push_back(raw_path[0]);
    reverse(new_path.begin(), new_path.end());

    return new_path;
}

void RRT::show_tree() {
    cout << "RRT Tree: " << endl;
    for (int i = 0; i < (int) tree.size(); ++i) {
        std::cout << "node #" << i << '\t'
                << "parent: #" << tree[i].parent << '\t'
                << std::setprecision(2) << std::fixed
                << "(" << tree[i].x << "," << tree[i].y << ")\t"
                << "cost: " << tree[i].cost << endl;
    }
    cout << endl << endl;
}

void RRT::show_path() {
    cout << "RRT Path: " << endl;
    for (int i = 0; i < (int) path.size(); ++i) {
        std::cout << "node #" << i << '\t'
                  << "parent: #" << path[i].parent << '\t'
                  << std::setprecision(2) << std::fixed
                  << "(" << path[i].x << "," << path[i].y << ")\t"
                  << "cost: " << path[i].cost << endl;
    }
    cout << endl << endl;
}

void RRT::show_smooth_path() {
    cout << "Smooth Path: " << endl;
    for (int i = 0; i < (int) smooth_path.size(); ++i) {
        std::cout << "node #" << i << '\t'
                  << "parent: #" << path[i].parent << '\t'
                  << std::setprecision(2) << std::fixed
                  << "(" << path[i].x << "," << path[i].y << ")\t"
                  << "cost: " << path[i].cost << endl;
    }
    cout << endl << endl;
}
