Point(1) = {0.0, 0.0, 0, 2};       // Bottom-left
Point(2) = {40.0, 0.0, 0, 2};      // Bottom-right
Point(3) = {40.0, 2.0, 0, 2};      // Top-right
Point(4) = {0.0, 2.0, 0, 2};       // Top-left

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};