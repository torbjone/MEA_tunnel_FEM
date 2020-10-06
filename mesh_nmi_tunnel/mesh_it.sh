# High resolution mesh by default 
gmsh -3 -optimize_netgen nmi_mea.geo -format msh2
dolfin-convert nmi_mea.msh nmi_mea.xml
