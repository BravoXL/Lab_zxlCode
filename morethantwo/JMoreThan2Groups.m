clc;
close all;
clear;
%目的：比较至少3组是否显著差异，第一步单因素方差分析，第二步，tukey检验。
%缺点，不好画出*，所以这个程序就仅仅用做统计计算吧

A = [20,22,29,21,18];
B = [15,14,16,13,17];
C = [10,12,11,9,13];
data = [A,B,C];
%lable the data,which number belongs to which group
group = [repmat({'A'},1,5),repmat({'B'},1,5),repmat({'C'},1,5)];
fprintf('---第一步、一起看有没有差异---------------------\n');
%
%第一步：一起比较看是否都一样？
%anova1 就是单因素方差分析。
%如果是on就会打开单因素方差分析的表
[p,tbl,stats]= anova1(data,group,'on');

fprintf('计算anova p:%.40f  。\n',p);
if p < 0.05
    fprintf('因为p<0.05，存在显著差异，你再去两两比较下。\n');
else
    fprintf('他们几组差不多');
end

fprintf('---第二步、做两两比较--用tukey-------------------\n');
%
% 第二步：两两比较看有无差别。tukey 多重比较检验（Multiple Comparison Test）

[c, m, h, gnames] = multcompare(stats, 'CType', 'tukey-kramer');

disp(array2table(c,'VariableNames',{'组1','组2','低CI','MeanDiff','高CI','p'}))
fprintf('zxl提示如何下结论：两两比较的p，如果p>0.05,二者基本相同。\n');


fprintf('---第三步、创建柱状图-------------------\n');
% 第三步：创建柱状图
means = [mean(A),mean(B),mean(C)];
errors = std([A,B,C],0,2) ./ sqrt(5);%标准误为标准差除以根号n
bar(means,'grouped');
hold on;

errorbar(1:3,means,errors,'k','linestyle','none');

for i = 1:size(c,1)
    if c(i,6)<0.05
        group1 = gnames{c(i,1)};
        group2 = gnames{c(i,2)};
        idx1 = find(strcmp(gnames,group1));
        idx2 = find(strcmp(gnames,group2));
        x = (idx1+idx2)/2;
        y = max(means)+max(errors)*0.01;
        text(x,y,'*','HorizontalAlignment','center','VerticalAlignment','bottom');
    end
end


legend('Group A','Group B','Group C');


title('Comparison of A,B and C');
xlabel('Group');
ylabel('Means');

set(gca,'XTickLabel',{'A','B','C'});

hold off;



