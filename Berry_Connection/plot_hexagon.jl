using PyPlot;
  
b_1 = 2*pi/3.0*[1, sqrt(3)]
b_2 = 2*pi/3.0*[1,-sqrt(3)]

K=[1,1/sqrt(3)]
K=K*2*pi/3

M=b_1/2.0


function rotate2D_vec(vec, angle_grad)
	angle=angle_grad/360*2.0*pi
        R=zeros(Float64,2,2)
	R[1,1]=cos(angle)
	R[2,1]=-sin(angle)
	R[1,2]=sin(angle)
	R[2,2]=cos(angle)
	rot_vec=R*vec
	return rot_vec
end

fig = figure("Hexagonal BZ",figsize=(10,10))
xlim(-4,4)
ylim(-4,4)

title("Hexagonal BZ");

quiver(0,0,b_1[1],b_1[2], units="xy",color="black", scale=1,width=0.025)
quiver(0,0,b_2[1],b_2[2], units="xy",color="black", scale=1,width=0.025)

for i in 1:6
	angle=(i-1)*60
	K_rot=rotate2D_vec(K,angle)
	M_rot=rotate2D_vec(M,angle)
	scatter(K_rot[1],K_rot[2], color="black")
	scatter(M_rot[1],M_rot[2], color="green")
end

xlabel("k_x [1/a units]");
ylabel("k_y [1/a units]");
  
PyPlot.show();
