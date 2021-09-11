function value = func1(x)
    load('func1', 'net');
    load('func1', 'inputps');
    load('func1', 'outputps');
    
    an = sim(net, mapminmax('apply',x,inputps));
    value = mapminmax('reverse',an,outputps);
end