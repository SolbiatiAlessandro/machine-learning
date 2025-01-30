# Value class starter code, with many functions taken out

import math

class Value:
  
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"


  def __gt__(self, other):
     if isinstance(other, Value):
       return self.data > other.data
     return self.data > other  

  def __truediv__(self, other): # self / other
    return self * (other ** -1)

  def log(self):
    assert self.data > 0, "Logarithm is only defined for positive values"
    out = Value(math.log(self.data), (self,), 'log')

    def _backward():
        # Derivative of log(x) is 1 / x
        self.grad += (1 / self.data) * out.grad

    out._backward = _backward
    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
        self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward

    return out
      
  def __add__(self, other): # exactly as in the video
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    
    return out

  def __sub__(self, other): # self - other
    return self + (-other)

  def __mul__(self, other):
      other = other if isinstance(other, Value) else Value(other)
      out = Value(self.data * other.data, (self, other), '*')

      def _backward():
          self.grad += other.data * out.grad
          other.grad += self.data * out.grad
      out._backward = _backward

      return out
      
  
  def exp(self):
      out = Value(math.exp(self.data), (self,), 'exp')

      def _backward():
          self.grad += out.data * out.grad
      out._backward = _backward
      return out

 
  def log(self):
    # Forward pass
    import math  # or from math import log at the top, then just call log(...)
    out = Value(math.log(self.data), (self,), 'log')
    
    def _backward():
        # derivative of log(x) wrt x is 1/x
        self.grad += (1.0 / self.data) * out.grad
    
    out._backward = _backward
    return out 

    
  def __neg__(self):
    out = Value(-self.data, (self,), 'neg')

    def _backward():
        self.grad += -out.grad  # derivative of -x is -1
    out._backward = _backward
    return out  

  def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

    def _backward():
        self.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out


  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')
    
    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out

  def sigmoid(self):
    x = self.data
    s = 1 / (1 + math.exp(-x))
    out = Value(s, (self, ), 'sigmoid')
    
    def _backward():
      self.grad += s * (1 - s) * out.grad

    out._backward = _backward
    return out
       
  def backward(self): # exactly as in video  
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()



from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot


import random

class Neuron:
  
  def __init__(self, nin, activation='relu'):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))
    self.activation = activation
  
  def __call__(self, x):
    # w * x + b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    if self.activation == 'relu':
        out = act.relu()
    elif self.activation == 'sigmoid':
        out = act.sigmoid()
    else:
      raise NotImplementedError
    return out
  
  def parameters(self):
    return self.w + [self.b]

class Layer:
  
  def __init__(self, nin, nout):
    activation = 'relu' if nout > 1 else 'sigmoid'
    self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]
  
  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
  
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

