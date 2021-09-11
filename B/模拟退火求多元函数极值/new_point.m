function newpoint = new_point(last_point, T)
    f = unifrnd(-1, 1)
    newpoint(1,1)  = last_point(1, 1) + unifrnd(100, 400) * T * 0.00005 * f;
    newpoint(2,1)  = last_point(2, 1) + unifrnd(0, 2) * T * 0.00005 * f;
    newpoint(3,1)  = last_point(3, 1) + unifrnd(0.5, 5) * T * 0.00005 * f;
	newpoint(4,1)  = last_point(4, 1) + unifrnd(0.3, 2.1) * T * 0.00005 * f;
    newpoint(5,1)  = last_point(5, 1) + unifrnd(250, 400) * T * 0.00005 * f;
end

