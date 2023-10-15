#ifndef RRT_H
#define RRT_H

#include <iostream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <random>

using namespace std;

/**
 * RRT Tree Node
 */
class TreeNode {
public:
    double x, y;
    double cost = 0.0;
    int parent = -1;
    bool is_root = false;
    bool close_to_goal = false;

    TreeNode() : x(0.0), y(0.0) {}

    TreeNode(double x, double y) : x(x), y(y) {}
};


/**
 * RRT Class
 */
class RRT {
public:
    RRT();

    RRT(const vector<double> &init,
        const vector<double> &goal,
        const vector<vector<vector<double>>> &occupancy_grid,
        double collision_tol,
        double expand_dis,
        double path_resolution,
        double goal_sample_rate,
        int max_iter,
        double circle_dist,
        bool early_stop,
        bool smooth);

    ~RRT() = default;

    void planning();

    void show_tree();

    void show_path();

    void show_smooth_path();

    // RRT Member Variables
    vector<TreeNode> tree;
    vector<TreeNode> path;
    vector<TreeNode> smooth_path;

private:
    // Random Generator
    mt19937 gen;
    uniform_real_distribution<> x_dist;
    uniform_real_distribution<> y_dist;
    uniform_real_distribution<> percent;

    // RRT Member Variables
    TreeNode start;
    TreeNode end;

    int num_occupied = 0;
    vector<double> occupied_x;
    vector<double> occupied_y;

    double xmin = 0.0;
    double xmax = 0.0;
    double ymin = 0.0;
    double ymax = 0.0;

    double collision_tol = 0.0;
    double expand_dis = 0.0;
    double path_resolution = 0.0;
    double goal_sample_rate = 0.0;
    int max_iter = 0;

    double circle_dist = 0.0;
    bool early_stop = false;

    bool smooth = false;

    // RRT Member Functions
    vector<double> sample();

    int nearest(const vector<double> &sampled_point);

    TreeNode steer(const TreeNode &nearest_node, vector<double> sampled_point) const;

    bool check_collision(const TreeNode &nearest_node, const TreeNode &new_node);

    bool is_goal(const TreeNode &latest_added_node, const TreeNode &goal_node);

    vector<TreeNode> find_path(const TreeNode &latest_added_node);

    double dist_to_grid(vector<double> point);

    int choose_parent(TreeNode &new_node, const vector<int> &neighborhood);

    void rewire(const TreeNode &new_node, const vector<int> &neighborhood);

    int search_best_goal_node();

    static double cost(const TreeNode &from_node, const TreeNode &to_node);

    static double line_cost(const TreeNode &n1, const TreeNode &n2);

    void propagate_cost_to_leaves(int parent);

    vector<int> near(const TreeNode &node);

    vector<TreeNode> path_smoother(const vector<TreeNode> &raw_path);
};

#endif //RRT_H