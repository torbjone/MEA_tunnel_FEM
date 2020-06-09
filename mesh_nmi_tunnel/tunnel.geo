
// Tunnel

pTu0 = newp; Point(pTu0) = {0, 0, 0, tunnel_mesh_size};

pTu1_left = newp; Point(pTu1_left) = {-structure_radius, -tunnel_thickness/2, 0, tunnel_mesh_size};
pTu2_left = newp; Point(pTu2_left) = {+structure_radius, -tunnel_thickness/2, 0, tunnel_mesh_size};
pTu3_left = newp; Point(pTu3_left) = {0, -structure_radius, 0, tunnel_cover_mesh_size};

pTu1_right = newp; Point(pTu1_right) = {-structure_radius, tunnel_thickness/2, 0, tunnel_mesh_size};
pTu2_right = newp; Point(pTu2_right) = {structure_radius, tunnel_thickness/2, 0, tunnel_mesh_size};
pTu3_right = newp; Point(pTu3_right) = {0, structure_radius, 0, tunnel_cover_mesh_size};


// Tunnel bottom lines
lTu0 = newl; Circle(lTu0) = {pTu1_left, pTu0, pTu3_left};
lTu1 = newl; Circle(lTu1) = {pTu3_left, pTu0, pTu2_left};
lTu2 = newl; Circle(lTu2) = {pTu1_right, pTu0, pTu3_right};
lTu3 = newl; Circle(lTu3) = {pTu3_right, pTu0, pTu2_right};


// Tunnel vertical lines

lTu4 = newl; Line(lTu4) = {pTu3_left, pCo4};
lTu5 = newl; Line(lTu5) = {pTu3_right, pCo2};

lTu6 = newl; Line(lTu6) = {pTu1_left, pCo1a};
lTu7 = newl; Line(lTu7) = {pTu2_left, pCo3a};
lTu8 = newl; Line(lTu8) = {pTu1_right, pCo1b};

lTu11 = newl; Line(lTu11) = {pTu2_right, pCo3b};

// Tunnel horizontal lines
lTu9 = newl; Line(lTu9) = {pTu1_left, pTu2_left};
lTu10 = newl; Line(lTu10) = {pTu1_right, pTu2_right};

lTu12 = newl; Line(lTu12) = {pTu1_left, pTu1_right};
lTu13 = newl; Line(lTu13) = {pTu2_left, pTu2_right};

llTu1 = newll; Line Loop(llTu1) = {lTu9, -lTu1, -lTu0};
llTu2 = newll; Line Loop(llTu2) = {lTu10, -lTu3, -lTu2};

llTu3 = newll; Line Loop(llTu3) = {lTu0, lTu4, lCo3, -lTu6};
llTu4 = newll; Line Loop(llTu4) = {lTu1, lTu7, lCo2, -lTu4};

llTu5 = newll; Line Loop(llTu5) = {lTu2, lTu5, -lCo0, -lTu8};
llTu6 = newll; Line Loop(llTu6) = {lTu3, lTu11, -lCo1, -lTu5};

llTu7 = newll; Line Loop(llTu7) = {lTu9, lTu7, -lCo12, -lTu6};
llTu8 = newll; Line Loop(llTu8) = {lTu10, lTu11, -lCo13, -lTu8};

llTu9 = newll; Line Loop(llTu9) = {lTu12, lTu8, -lCo3a, -lTu6};
llTu10 = newll; Line Loop(llTu10) = {lTu13, lTu11, -lCo2a, -lTu7};

ll_st_bottom = newll; Line Loop(ll_st_bottom) = {lTu0, lTu1, lTu13, -lTu3, -lTu2, -lTu12};
llTu_bottom = newll; Line Loop(llTu_bottom) = {lTu9, lTu13, -lTu10, -lTu12};

sTuVert1 = news; Surface(sTuVert1) =  {llTu3};
sTuVert2 = news; Surface(sTuVert2) =  {llTu4};
sTuVert3 = news; Surface(sTuVert3) =  {llTu5};
sTuVert4 = news; Surface(sTuVert4) =  {llTu6};
sTuVert5 = news; Surface(sTuVert5) =  {llTu7};
sTuVert6 = news; Surface(sTuVert6) =  {llTu8};

sTuVert7 = news; Surface(sTuVert7) =  {llTu9};
sTuVert8 = news; Surface(sTuVert8) =  {llTu10};

sTuBottom = news; Surface(sTuBottom) = {llTu_bottom};

//sTuBottomLeft = news; Surface(sTuBottomLeft) =  {llTu1};
//sTuBottomRight = news; Surface(sTuBottomRight) =  {llTu2};

