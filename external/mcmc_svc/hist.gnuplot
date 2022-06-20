binwidth=10
bin(x,width)=width*floor(x/width)

set xrange [0 : 1000];
plot 'history_N2.txt' using (bin($1,binwidth)):(1.0) smooth freq with boxes
