clear
clc
close
training_samples=300;
y=(sin(linspace(0,10*pi,800)).*cos((linspace(0,40*pi,800))))';
y_train=con2seq(y(1:training_samples)');
y_vaild=con2seq(y(training_samples+1:end)');
w=con2seq(ones(size(y(training_samples+1:end)))');
inputDelays=1:6:19;
hiddensizes=[6 3];
net=narnet(inputDelays,hiddensizes);
[Xs,Xi,Ai,Ts]=preparets(net,{},{},y_train);
net=train(net,Xs,Ts,Xi,Ai);
view(net)
net=closeloop(net);
view(net)
yini=y_train(end-max(inputDelays)+1:end);
[Xs,Xi,Ai]=preparets(net,{},{},[yini w]);
predict=net(Xs,Xi,Ai);
Yv=cell2mat(y_vaild);
Yp=cell2mat(predict);
e=Yv-Yp;
N=800;
hold on
plot(training_samples+1:N,Yv,'--k')
plot(training_samples+1:N,Yp,'r')
plot(training_samples+1:N,e,'g')
legend('validation data','training data','error')
