function value = func2(x)
    load('func2', 'net');
    load('func2', 'inputps');
    load('func2', 'outputps');
    
    an = sim(net, mapminmax('apply',x,inputps));
    value = mapminmax('reverse',an,outputps);
end