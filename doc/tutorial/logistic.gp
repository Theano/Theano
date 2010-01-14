set terminal svg font "Bitstream Vera Sans,10" size 300,200
set output "logistic.svg"

set xrange [-6:6]
set xzeroaxis linetype -1
set yzeroaxis linetype -1
set xtics axis nomirror
set ytics axis nomirror 0,0.5,1
set key off
set grid
set border 1

set samples 400

plot 1/(1 + exp(-x)) with line linetype rgbcolor "blue" linewidth 2

set ytics axis nomirror 0,0.25
set output "dlogistic.svg"
plot 1/(1 + exp(-x)) * (1 - 1/(1 + exp(-x))) with line linetype rgbcolor "blue" linewidth 2
