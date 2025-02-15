Point(1) = {100.0, 50.0, 0, 2};
Point(2) = {0.0, 50.0, 0, 2};
Point(3) = {100.0, 24.0, 0, 2};
Point(4) = {100.0, 26.0, 0, 2};
Point(5) = {100.0, 0.0, 0, 2};
Point(6) = {0.0, 0.0, 0, 2};
Point(7) = {0.0, 26.0, 0, 2};
Point(8) = {0.0, 24.0, 0, 2};
//+
Line(1) = {6, 8};
//+
Line(2) = {8, 7};
//+
Line(3) = {7, 2};
//+
Line(4) = {2, 1};
//+
Line(5) = {1, 4};
//+
Line(6) = {4, 3};
//+
Line(7) = {3, 5};
//+
Line(8) = {5, 6};
//+
Line(9) = {8, 3};
//+
Line(10) = {7, 4};
//+
Curve Loop(1) = {3, 4, 5, -10};
//+
Curve Loop(2) = {1, 9, 7, 8};
//+
Plane Surface(1) = {1};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {2, 10, 6, -9};
//+
Plane Surface(3) = {3};
//+
Field[1] = Box;
//+
Field[1].VIn = 0.15;
//+
Field[1].VOut = 10;
//+
Field[1].XMax = 56;
//+
Field[1].XMin = 44;
//+
Field[1].YMax = 31;
//+
Field[1].YMin = 19;
//+
Background Field = 1;
//+
Field[1].VOut = 50;
//+
Field[1].VOut = 1;
//+
Delete Field [1];
