mesh_outer = 50;
tunnel_cover_mesh_size = 2;
tunnel_mesh_size = 0.25;

cylinder_radius = 500;

structure_radius = 100;
cover_height = 50;

cylinder_height = cylinder_radius;

tunnel_thickness = 5;
tunnel_height = 5;


// Cylinder
Include "outer_cylinder.geo";

// Tunnel
Include "tunnel_cover.geo";
Include "tunnel.geo";

sCyLower = news; Surface(sCyLower) = {llCyLower, ll_st_bottom};


slTu = newsl; Surface Loop(slTu) = {sTuBottom, sTuVert7, sTuVert8, sTuVert5, sTuVert6, sTunnelTop};
vTu = newv; Volume(vTu) = {slTu};

sl1 = newsl; Surface Loop(sl1) = {sCyVert1, sCyVert2, sCyVert3, sCyVert4, sCyUpper, sCyLower, sTuVert1, sTuVert2, sTuVert3, sTuVert4, sTuVert7, sTuVert8, sCoVert1, sCoVert2, sCoVert3, sCoVert4, sCoVert5, sCoVert6, sCoUpper};

v2 = newv; Volume(v2) = {sl1};

Physical Volume(1) = {v2, vTu};
Physical Surface(1) = {sCyVert1, sCyVert2, sCyVert3, sCyVert4, sCyUpper};
