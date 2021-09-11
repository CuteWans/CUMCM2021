function newpoint = new_point(~)%生成新点的函数
    newpoint(1,1)  = unifrnd(100, 400);
    newpoint(2,1)  = unifrnd(0.5, 5);
    newpoint(3,1)  = unifrnd(0.8, 3.6);
	newpoint(4,1)  = unifrnd(0.3, 2.1);
    newpoint(5,1)  = unifrnd(250, 400);
end