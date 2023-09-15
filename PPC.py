
#    Project Name: Autonomous Vehicle Control in Unity
#    Description: This C# script is part of the Autonomous Vehicle Control in Unity project. 
#                 The aim of the project is to controls a vehicle from it's start to the selected target destination by avoiding obstacles.
#                 It includes path following, speed adjustment, and data logging. It integrates various components such as
#                 AStar pathfinding, Pure Pursuit Control and PID controller.
#    Author: [Alessandro Assirelli]
#    Date: [Project Course 2022-2023]
#    Version: 1.0

import numpy as np


class PPC :

    def __init__(self):
        self.lastFoundIndex = 0 # Stores the index of the last segment the intersection has been found on
        self.lookAheadDistance = 1; # Distance to look ahead for path intersection

    # Compute the steering angle for path following
    def ComputeSteeringAngle(self, path, currentPosition, L):
        # Parameters:
        # - path: List of waypoints (in {x,z} world coordinates) representing the path the vehicle should follow.
        # - currentPosition: The current position and orientation of the vehicle.
        # - L: The wheelbase of the vehicle (distance between front and rear axles).

        currentGoal = ComputeIntersection(path, currentPosition.position)
        turn_error = ComputeTurnError(currentGoal, currentPosition)
        alpha = -(Mathf.Atan(2 * Mathf.Sin(turn_error) * L / lookAheadDistance)) * 180 / Mathf.PI
        return alpha

    # Compute the steering angles for both left and right wheels (for vehicles with Ackermann steering)
    # public List<float> ComputeSteeringAngle(List<Vector3> path, Transform currentPosition, float L, float interaxialDistance)
    # {
    #     # Parameters:
    #     # - path: List of waypoints (in {x,z} world coordinates) representing the path the vehicle should follow.
    #     # - currentPosition: The current position and orientation of the vehicle.
    #     # - L: The wheelbase of the vehicle (distance between front and rear axles).
    #     # - interaxialDistance: The distance between left and right wheels.

    #     var currentGoal = ComputeIntersection(path, currentPosition.position);
    #     var turn_error = ComputeTurnError(currentGoal, currentPosition);
    #     var alpha = -(Mathf.Atan(2 * Mathf.Sin(turn_error) * L / lookAheadDistance)) * 180 / Mathf.PI;
    #     var R = lookAheadDistance / (2 * Mathf.Sin(turn_error));
    #     var alpha_left = -(Mathf.Atan(L / (R - interaxialDistance / 2))) * 180 / Mathf.PI;
    #     var alpha_right = -(Mathf.Atan(L / (R + interaxialDistance / 2))) * 180 / Mathf.PI;
    #     List<float> result = new List<float>() { alpha_left, alpha_right };
    #     return result;
    # }

    # Compute the intersection between the vehicle's path and a lookahead distance
    def ComputeIntersection(self, path, currentPos):
        # Parameters:
        # - path: List of waypoints representing the path the vehicle should follow.
        # - currentPos: Current position of the vehicle in world coordinates.

        intersection = self.FindGoalPt(path, currentPos, self.lookAheadDistance, lastFoundIndex)
        currentGoal = intersection[0]
        lastFoundIndex = intersection[1]
        return currentGoal

    # Compute the error in turning towards a target position
    def ComputeTurnError(self, currentGoal, currentPosition):
        # Parameters:
        # - currentGoal: The target position the vehicle is trying to reach.
        # - currentPosition: The current position and orientation of the vehicle.

        angle_to_reach = np.arctan2(currentGoal.z - currentPosition.position.z, currentGoal.x - currentPosition.position.x)
        current_heading = np.arctan2(currentPosition.forward.z, currentPosition.forward.x)
        turn_error = (angle_to_reach - current_heading)
        return turn_error
    

    # Reset the last found index
    def ResetPPC(self):
        # Resets the index used for finding the last goal point on the path.
        self.lastFoundIndex = 0

    # Find the goal point on the path given the lookahead distance
    def FindGoalPt(self, path, current):
        # Parameters:
        # - path: List of waypoints representing the path the vehicle should follow.
        # - current: Current position of the vehicle in world coordinates.
        # - Ld: Lookahead distance for finding the goal point.
        # - lastFoundIndex: Index of the last found segment on the path.

        goal_pt = np.array([0, 0])
        nsol = 0
        startingIndex = self.lastFoundIndex
        terminal = False

        for i in range(startingIndex, len(path) - 1):
        
            # Tolerance for checking if a point is on the path
            TOL = 1E-5
            x1 = path[i][0]
            x2 = path[i + 1][0]
            z1 = path[i][1]
            z2 = path[i + 1][1]

            # Maximum and minimum values for x and z coordinates
            maxX = np.max(x1, x2) + TOL
            minX = np.min(x1, x2) - TOL
            maxZ = np.max(z1, z2) + TOL
            minZ = np.min(z1, z2) - TOL

            x1_center = x1 - current[0]
            x2_center = x2 - current[0]
            z1_center = z1 - current[1]
            z2_center = z2 - current[1]

            dx = x2_center - x1_center
            dz = z2_center - z1_center
            dr = np.sqrt(pow(dx, 2) + pow(dz, 2))
            D = x1_center * z2_center - x2_center * z1_center

            Delta = pow(self.lookAheadDistance, 2) * pow(dr, 2) - pow(D, 2)

            if (Delta >= 0):
                x1_sol = current[0] + (D * dz + np.sign(dz) * dx * np.sqrt(Delta)) / pow(dr, 2)
                x2_sol = current[0] + (D * dz - np.sign(dz) * dx * np.sqrt(Delta)) / pow(dr, 2)
                z1_sol = current[1] + (-D * dx + np.abs(dz) * np.sqrt(Delta)) / pow(dr, 2)
                z2_sol = current[1] + (-D * dx - np.abs(dz) * np.sqrt(Delta)) / pow(dr, 2)

                # Check if the first solution is in range
                if (minX <= x1_sol and x1_sol <= maxX and minZ <= z1_sol and z1_sol <= maxZ) or (minX <= x2_sol and x2_sol <= maxX and minZ <= z2_sol and z2_sol <= maxZ):
                    if (minX <= x1_sol and x1_sol <= maxX and minZ <= z1_sol and z1_sol <= maxZ):

                        goal_pt[0] = x1_sol
                        goal_pt[1] = z1_sol
                        nsol += 1
                    
                    # Check if the second solution is in range
                    if (minX <= x2_sol and x2_sol <= maxX and minZ <= z2_sol and z2_sol <= maxZ):

                        goal_pt[0] = x2_sol
                        goal_pt[1] = z2_sol
                        nsol += 1

                    # If both solutions are valid, select the closest to the second point
                    if (nsol == 2):
                        dr_sol1 = np.sqrt(pow(x2 - x1_sol, 2) + pow(z2 - z1_sol, 2))
                        dr_sol2 = np.sqrt(pow(x2 - x2_sol, 2) + pow(z2 - z2_sol, 2))

                        if (dr_sol1 < dr_sol2):
                        
                            goal_pt[0] = x1_sol
                            goal_pt[1] = z1_sol
                        
                        else:
                        
                            goal_pt[0] = x2_sol
                            goal_pt[1] = z2_sol
                        
                    # Check if the goal point is closer to the current position than the next path point
                    if np.sqrt( (goal_pt[0] - current[0]) ** 2 + (goal_pt[1] - current[1]) ** 2)  <= np.sqrt( (path[0, i+1] - current[0]) ** 2 + (path[i+1, 1] - current[1]) ** 2):
                    
                        self.lastFoundIndex = i
                        break
                    
                    else:
                    
                        self.lastFoundIndex = i + 1
                        if (self.lastFoundIndex == len(path) - 1):
                        
                            terminal = True
        

        if (nsol == 0):
        
            goal_pt = path[self.lastFoundIndex]

            if (self.lastFoundIndex == len(path) - 2):
            
                goal_pt = path[len(path) - 1]
            
    

        if (terminal or self.lastFoundIndex == len(path) - 1):
        
            goal_pt = path[len(path) - 1]
        
        return goal_pt
    
