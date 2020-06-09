# High resolution mesh by default 
# For low res change sphere_4.geo to sphere_4_lowres.geo and sphere_4.msh to sphere_4_lowres.msh
gmsh -3 -optimize_netgen nmi_mea.geo -format msh2
dolfin-convert nmi_mea.msh nmi_mea.xml
