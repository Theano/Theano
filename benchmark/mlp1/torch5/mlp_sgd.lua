
require "lab"
require "os"
dataset={};
n_examples=1000;
inputs=1000;
outputs=12;
HUs=500;

function dataset:size() return n_examples end -- 100 examples
for i=1,dataset:size() do 
  local input = lab.randn(inputs);     -- normally distributed example in 2d
  local output = lab.randn(outputs);
  dataset[i] = {input, output}
end

require "nn"
mlp = nn.Sequential();  -- make a multi-layer perceptron
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs))


criterion = nn.MSECriterion()  
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer.shuffleIndices = true
trainer.maxIteration = 4
trainer:train(dataset)

