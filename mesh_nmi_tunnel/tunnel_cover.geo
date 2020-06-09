
// Tunnel cover

pCo0 = newp; Point(pCo0) = {0, 0, tunnel_height, tunnel_mesh_size};

pCo1a = newp; Point(pCo1a) = {-structure_radius, -tunnel_thickness/2, tunnel_height, tunnel_mesh_size};
pCo1b = newp; Point(pCo1b) = {-structure_radius, tunnel_thickness/2, tunnel_height, tunnel_mesh_size};

pCo2 = newp; Point(pCo2) = {0, structure_radius, tunnel_height, tunnel_cover_mesh_size};

pCo3a = newp; Point(pCo3a) = {+structure_radius, -tunnel_thickness/2, tunnel_height, tunnel_mesh_size};
pCo3b = newp; Point(pCo3b) = {+structure_radius, +tunnel_thickness/2, tunnel_height, tunnel_mesh_size};

pCo4 = newp; Point(pCo4) = {0, -structure_radius, tunnel_height, tunnel_cover_mesh_size};
pCo5 = newp; Point(pCo5) = {0, 0, tunnel_height + cover_height, tunnel_cover_mesh_size};

pCo6a = newp; Point(pCo6a) = {-structure_radius, -tunnel_thickness/2, tunnel_height + cover_height, tunnel_cover_mesh_size};
pCo6b = newp; Point(pCo6b) = {-structure_radius, +tunnel_thickness/2, tunnel_height + cover_height, tunnel_cover_mesh_size};

pCo7 = newp; Point(pCo7) = {0, structure_radius, tunnel_height + cover_height, tunnel_cover_mesh_size};

//pCo8 = newp; Point(pCo8) = {+structure_radius, 0, tunnel_height + cover_height, tunnel_cover_mesh_size};

pCo8a = newp; Point(pCo8a) = {+structure_radius, -tunnel_thickness/2, tunnel_height + cover_height, tunnel_cover_mesh_size};
pCo8b = newp; Point(pCo8b) = {+structure_radius, +tunnel_thickness/2, tunnel_height + cover_height, tunnel_cover_mesh_size};

pCo9 = newp; Point(pCo9) = {0, -structure_radius, tunnel_height + cover_height, tunnel_cover_mesh_size};

// Cylinder bottom lines
lCo0 = newl; Circle(lCo0) = {pCo1b, pCo0, pCo2};
lCo1 = newl; Circle(lCo1) = {pCo2, pCo0, pCo3b};
lCo2 = newl; Circle(lCo2) = {pCo3a, pCo0, pCo4};
lCo2a = newl; Circle(lCo2a) = {pCo3a, pCo0, pCo3b};

lCo3 = newl; Circle(lCo3) = {pCo4, pCo0, pCo1a};
lCo3a = newl; Circle(lCo3a) = {pCo1a, pCo0, pCo1b};

// Cylinder top lines
lCo4 = newl; Circle(lCo4) = {pCo6b, pCo5, pCo7};
lCo5 = newl; Circle(lCo5) = {pCo7, pCo5, pCo8b};
lCo6 = newl; Circle(lCo6) = {pCo8a, pCo5, pCo9};

lCo6a = newl; Circle(lCo6a) = {pCo8a, pCo5, pCo8b};
lCo7 = newl; Circle(lCo7) = {pCo9, pCo5, pCo6a};

lCo7a = newl; Circle(lCo7a) = {pCo6a, pCo5, pCo6b};

// Cylinder vertical lines
lCo8a = newl; Line(lCo8a) = {pCo1a, pCo6a};
lCo8b = newl; Line(lCo8b) = {pCo1b, pCo6b};
lCo9 = newl; Line(lCo9) = {pCo2, pCo7};
lCo10a = newl; Line(lCo10a) = {pCo3a, pCo8a};
lCo10b = newl; Line(lCo10b) = {pCo3b, pCo8b};
lCo11 = newl; Line(lCo11) = {pCo4, pCo9};

// Cylinder horizontal lines
lCo12 = newl; Line(lCo12) = {pCo1a, pCo3a};
lCo13 = newl; Line(lCo13) = {pCo1b, pCo3b};

llCoTunnelTop = newll; Line Loop(llCoTunnelTop) = {lCo12, lCo2a, -lCo13, -lCo3a};

llCoUpper = newll; Line Loop(llCoUpper) = {lCo4,lCo5, -lCo6a, lCo6,lCo7, lCo7a};
llCoLower = newll; Line Loop(llCoLower) = {lCo0,lCo1, -lCo2a, lCo2,lCo3, lCo3a};

llCoVert1 = newll; Line Loop(llCoVert1) = {lCo0, lCo9, -lCo4, -lCo8b};
llCoVert2 = newll; Line Loop(llCoVert2) = {lCo1, lCo10b, -lCo5, -lCo9};
llCoVert3 = newll; Line Loop(llCoVert3) = {lCo2, lCo11, -lCo6, -lCo10a};
llCoVert4 = newll; Line Loop(llCoVert4) = {lCo3, lCo8a, -lCo7, -lCo11};

llCoVert5 = newll; Line Loop(llCoVert5) = {lCo8a, lCo7a, -lCo8b, -lCo3a};
llCoVert6 = newll; Line Loop(llCoVert6) = {lCo10a, lCo6a, -lCo10b, -lCo2a};

sCoVert1 = news; Surface(sCoVert1) =  {llCoVert1};
sCoVert2 = news; Surface(sCoVert2) =  {llCoVert2};
sCoVert3 = news; Surface(sCoVert3) =  {llCoVert3};
sCoVert4 = news; Surface(sCoVert4) =  {llCoVert4};

sCoVert5 = news; Surface(sCoVert5) =  {llCoVert5};
sCoVert6 = news; Surface(sCoVert6) =  {llCoVert6};

sCoUpper = news; Plane Surface(sCoUpper) = {llCoUpper};
//sCoLower = news; Plane Surface(sCoLower) = {llCoLower};
sTunnelTop = news; Surface(sTunnelTop) = {llCoTunnelTop};
