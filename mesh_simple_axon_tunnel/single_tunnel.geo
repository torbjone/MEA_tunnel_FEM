size = 1;

// Box spec
x0 = -100;
y0 = -2.5;
z0 = 0;
dx = 200;
dy = 5;
dz = 5;

// Rectangle
//
//   8----------7
//  /|         /| 
// 5----------6 |
// | |        | |
// | 4--------|-3           /y
// |/         |/           /
// 1----------2            ------
//                              x 


// Base
Point(1) = {x0, y0, z0, size};
Point(2) = {x0+dx, y0, z0, size};
Point(3) = {x0+dx, y0+dy, z0, size};
Point(4) = {x0, y0+dy, z0, size};

Line(12) = {1, 2};
Line(23) = {2, 3};
Line(34) = {3, 4};
Line(41) = {4, 1};

// Shifted Base
Point(5) = {x0, y0, z0+dz, size};
Point(6) = {x0+dx, y0, z0+dz, size};
Point(7) = {x0+dx, y0+dy, z0+dz, size};
Point(8) = {x0, y0+dy, z0+dz, size};


// Verticals
Line(15) = {1, 5};
Line(26) = {2, 6};
Line(37) = {3, 7};
Line(48) = {4, 8};

// Top lines
Line(56) = {5, 6};
Line(67) = {6, 7};
Line(78) = {7, 8};
Line(85) = {8, 5};


Line Loop(1) = {12, 23, 34, 41};  // Might need curve loop here instead
Plane Surface(1) = {1};

Line Loop(2) = {26, 67, -37, -23};
Plane Surface(2) = {2};


Line Loop(3) = {34, 48, -78, -37};
Plane Surface(3) = {3};

Line Loop(4) = {48, 85, -15, -41};
Plane Surface(4) = {4};

Line Loop(5) = {15, 56, -26, -12};
Plane Surface(5) = {5};

Line Loop(6) = {56, 67, 78, 85};
Plane Surface(6) = {6};


Surface Loop(1) = {2, 3, 5, 4, 6, 1};
Volume(1) = {1};

Physical Surface(1) = {1}; // Bottom (MEA)

Physical Volume(1) = {1};
