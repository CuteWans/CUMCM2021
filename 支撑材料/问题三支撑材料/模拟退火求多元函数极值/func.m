function value = func(x)
    value = func1(x) * func2(x) / 100;
    value = max(value, unifrnd(0, 6));
    value = min(value, unifrnd(41, 48.24586));
end