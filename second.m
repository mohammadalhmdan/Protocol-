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