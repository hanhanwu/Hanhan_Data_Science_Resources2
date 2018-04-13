library(DiagrammeR)

# visualize a transition system
## The shape here controls the shape of each state
grViz("
      digraph markov_chain {
        graph[overlap=true, fontsize=10, rankdir=LR]
        node[shape=circle] 0;1;
        0->0 0->1 1->0 1->1
      }", height=125)
