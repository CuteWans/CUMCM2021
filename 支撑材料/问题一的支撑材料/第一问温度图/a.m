[mydata] = textread('input.txt');
data = mydata
l = 1; r = 2; cnt = 1;
while l <= 114
    while r < 114 && data(r, 1) < data(r + 1, 1)
        r = r + 1;
    end
    temp = data(l:r, 1);
    y1 = data(l:r, 2);
    y2 = data(l:r, 3);
    % figure(floor((cnt - 1) / 3) + 1)
    if cnt > 1
        figure(2)
        subplot(4, 5, cnt - 1)
    else
        figure(1)
    end
    % myfigure = figure(cnt)
    plot(temp, y1, 'r');
    hold on
    plot(temp, y2, 'b');
    hold on
    if cnt <= 14
        str = sprintf("A" + num2str(cnt));
        title(str)
    else
        str = sprintf("B" + num2str(cnt - 14));
        title(str)
    end
    str_t = sprintf('温度\n');
    xlabel(str_t)
    % axis([250, 400, 0, 100])
    if cnt == 1
        legend('乙醇转化率', 'C4烯烃选择性')
    end
    % saveas(myfigure, num2str(cnt), 'png');
    l = r + 1;
    r = l + 1;
    cnt = cnt + 1;
end