clear
clc
close
training_samples=300;
y=(sin(linspace(0,10*pi,800)).*cos((linspace(0,40*pi,800))))';
plot(y,'m-')
grid on , hold on
plot(y(1:training_samples),'b')
plot(y,'+k','markersize',2)
legend('validation data','training data','sampling markers')

xlabel('time (steps)')
ylabel('y')