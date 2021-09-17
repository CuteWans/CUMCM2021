[input_text] = textread('input.txt');
input = input_text;
an = zeros(22, 1);
for i = 1:22
    an(i, 1) = sum(input(i, :).*input(22, :));
    an(i, 1) = an(i, 1) ./ sqrt(sum(input(i, :).^2));
    an(i, 1) = an(i, 1) ./ sqrt(sum(input(22, :).^2));
end
an