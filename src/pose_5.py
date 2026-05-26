import numpy as np
import pickle
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def copy_graph_and_estimate(graph, initial_estimate):
    return pickle.loads(pickle.dumps(graph)), pickle.loads(pickle.dumps(initial_estimate))

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer 
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)

    # TODO: Perform the optimization and print the result
    result = optimizer.optimize()

    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    best_pose = None      # chosen pose option
    best_landmark = None    # chosen landmark (1 or 2)
    best_sum_of_marginals = float("inf")
    best_total_marginals = float("inf")

    # TODO: Calculate marginal covariances for the relevant variables and visualize the updated factor graph with covariances
    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            trial_graph, trial_estimate = copy_graph_and_estimate(graph, initial_estimate)
            trial_graph, trial_estimate = add_pose(trial_graph, trial_estimate, pose_5)
            result = optimize(trial_graph, trial_estimate)
            trial_graph = add_landmark_measurement(trial_graph, result, pose_5, landmark)
            result = optimize(trial_graph, trial_estimate)
            marginals = gtsam.Marginals(trial_graph, result)
            sum_of_marginals = (
                marginals.marginalCovariance(L(1)).sum()
                + marginals.marginalCovariance(L(2)).sum()
            )
            selected_landmark_marginal = marginals.marginalCovariance(L(landmark)).sum()
            if selected_landmark_marginal < best_sum_of_marginals:
                best_pose = pose_key
                best_landmark = landmark
                best_sum_of_marginals = selected_landmark_marginal
                best_total_marginals = sum_of_marginals

    # The sum of the marginals for each landmark can be computed using marginals.marginalCovariance(L(x)).sum()
    return best_pose, best_landmark, best_total_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.
    best_pose = None      # chosen pose option
    best_landmark = None    # chosen landmark (1 or 2)
    best_error = float("inf")
    best_pose_error = float("inf")

    # TODO: create a list of errors (each index corresponds to a pose) and add the error of each pose to the list
    list_of_errors = []
    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            trial_graph, trial_estimate = copy_graph_and_estimate(graph, initial_estimate)
            trial_graph, trial_estimate = add_pose(trial_graph, trial_estimate, pose_5)
            result = optimize(trial_graph, trial_estimate)
            trial_graph = add_landmark_measurement(trial_graph, result, pose_5, landmark)
            candidate_error = trial_graph.error(trial_estimate)
            result = optimize(trial_graph, trial_estimate)
            error = 0
            correct_poses = {
                1: gtsam.Pose2(0.0, 0.0, 0.0),
                2: gtsam.Pose2(2.0, 0.0, 0.0),
                3: gtsam.Pose2(4.0, 0.0, 0.0),
            }
            for pose_number, correct_pose in correct_poses.items():
                optimized_pose = result.atPose2(X(pose_number))
                error += abs(optimized_pose.x() - correct_pose.x())
                error += abs(optimized_pose.y() - correct_pose.y())
                error += abs(optimized_pose.theta() - correct_pose.theta())
            list_of_errors.append(error)
            if candidate_error < best_error:
                best_pose = pose_key
                best_landmark = landmark
                best_error = candidate_error
                best_pose_error = error

    # TODO: compute the sum of the errors and return it along with the best pose and landmark
    sum_of_errors = best_pose_error
    return best_pose, best_landmark, sum_of_errors 
