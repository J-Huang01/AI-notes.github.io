# CSCI 4511w

### Introduction to Artificial Intelligence

## 1/18/2023 Lecture 1

### What is AI?

#### Definition:

1. Acting humanly

   Turing test

   chatbots

2. Thinking Humanly

   Model how brain works

   Do that

3. Thinking Rationally

   Logic - Rules of thought

   Drow conclusions based on input

4. Acting Rationally

   Do the correct things

   "Correctness" defined by some **mathematical formula**

### Rational Agents

Agents

f: Perception -> Action

Perfect rationality is often unachievable

Time, Space, Knowledge gaps

### AI or not? To what extent?

- Bar code scanner(store) X
  - phone QR code
  - Maybe AI for vision
  - tabel look up √
- Web search engine
  - Graph + Connections
  - Contextual
  - use other info
- Thermostat X
- Voice-Response telephone menus
- Path-Planning systems respond to traffic conditions

### Agent acting in environment

- "Program to solve problem"

### Agent interaction

- Sencors to perceive environment

- Actnators to act on environment

## 1/23/2023 Lecture 2

> Definition &
>
> a basic environment
>
> Rationality, Task environments
>
> - examples
>
> Environment Properties
>
> - in class exercise
>
> Agent

#### Definitions

- Agent
  - Perceive (via sensors) and act in environment

- Percept 
  - content that **sensor's are perceiving** at a single moment
- Percept Sequence
  - Complete **history** of an agent's percepts

- State 

  - Uique configuration of agent in its environment

- Example - Vacuum World

  - 2 locations A B
  - each location can be 
    - clean
    - dirty
  - Actions 
    - Left
    - right
    - suck
    - no option
  - sensors 
    - Location
    - Clean/dirty

- Percept

  - [A, clean]
  - [A, dirty]
  - [B, clean]
  - [B, dirty]

- Plan to clean all

  > if dirty -> suck
  >
  > Then if A -> right         
  >
  > ​          If B -> left

- Performance measure

  - 1000 time steps
  - +1 for each clean location per time step
  - -0.5 per action
  - 0 for no option

### PEAS

![截屏2023-01-23 16.38.52](/Users/huangjiatan/Library/Application Support/typora-user-images/截屏2023-01-23 16.38.52.png)

- Playing Tennis

  - Performance Measure

    - Hit ball/not

    - In/Hitmet/...

    - Win +100

    - Location of ball

    - Rules?

    - 6000 serve vs not

    - Game **win +1, Loss -1**

- Perform a high jump

  - Performance Measure
    - Jump higher than bar +1
    - Didn't 0
    - Height in cm

- knitting a sweater

  - Performance Measure
    - % finish
    - Success in replication
    - HEAT conservation
    - Time
    - Strength

- Bidding at Auction

#### Properties of Task Environment

- Observability
  - Fully - board (play chess)
    - Agent sensors can "see" everything
  - Partially
    - not
- Agent Number
  - Single 
  - Multi (more than 1)
- Determinism
  - If the next state of the environment is completely determined by the current state and action - **Deterministic**
  - **Deterministic**: We can predict environment state after acting
  - **Nondeterministic**: Not
- Static vs. Dynamic
  - Static: Environment doesn't change while agent is "thinking"
  - Dynamic: Not
  - Semi-Dynamic: Performance measure can change while "thinking"
- Discrete vs. Continuous
  - Discrete: Nature of time & space in task environment
  - Continuous

## 1/25/2023 Lecture 3

> In class exercise answer
>
> Task environment properties
>
> Agent Structre
>
> Simple task environments



> Q1 Vacuum World
>
> Consider the vacuum-cleaner world from the book (figure 2.2). This world has **two locations (A and B)**. Each location can be **clean or dirty**. The agent can move left or right and clean the location that it occupies. The performance measure for this world is as follows: 1 point per clean square at each step, over 1000 steps, at which time the agent stops.
>
> How many possible *states* (unique configurations of the agent in its environment) exist?
>
> - State: unique configuration of agent in environment
> - 2(agent) * 2(location) * 2 = 8
>
> What is the maximum length of a *percept sequence* for this world?
>
> - 1000



|                         | OBS     | Agent# | DET  | STAT/DYN          | DISC/CONT  |
| ----------------------- | ------- | ------ | ---- | ----------------- | ---------- |
| Crossword puzzle solver | Fully   | Single | DET  | Static            | Discrete   |
| Poker player            | Partial | Multi  | Non  | Static (no clock) | Discrete   |
| Self-driving car        | Partial | Multi  | Non  | Dynamic           | Continuous |
| Chess with clock        | Fully   | Multi  | DET  | Semi-Dynamic      | Discrete   |

Multi: could be AI play with a human

#### Agent Structure

- Simple Reflex Agent
  
  - ![截屏2023-01-30 18.37.11](/Users/huangjiatan/Library/Application Support/typora-user-images/截屏2023-01-30 18.37.11.png)
  
  - It is exactly a lookup table. Agent has a sequence of if-then statements to react to percept sequence.
  
- Model-Based Reflex Agent
  
  - ![截屏2023-01-30 18.37.33](/Users/huangjiatan/Library/Application Support/typora-user-images/截屏2023-01-30 18.37.33.png)
  
  - Agent maintains a model of the world.
  - This model can represent information that may not be currently directly detectable by sensors.
  - Model is used as lookup into table to choose action. 
  
- Goal-Based Agent
  
  - ![截屏2023-01-30 18.37.54](/Users/huangjiatan/Library/Application Support/typora-user-images/截屏2023-01-30 18.37.54.png)
  
  - Agent maintains a model of the world.
  - This model can represent information that may not be currently directly detectable by sensors.
  - Agent is aware of how its actions affect the world.
  - Agent has a “goal”, a set of information that describes desirable situations.
  - Agent chooses actions that will move it closer to a goal.

#### Puzzles (easy environment)

Fully Obs 

Single Agent 

Det 

Static 

Discrete

#### Solving puzzles with search

Define problem

- Initial state
- Goal test
- Actions for a given state
  - Function action(s)
- Description of results of agents
  - "Transition model"
  - Function -> Result(a, s)
    - Returns the state we end up in if we're in state s and take action a
- Path cost
  - Function that assigns numeric value to get from one state to another

Initial state $s_0$

three possible options $a_1$, $a_2$, $a_3$

 $s_1$,  $s_2$,  $s_3$

...g(this is the goal)

## 1/30/2023 Lecture 4

> - hw01
>
> - Problem solving process
>   - Definition
>   - Example
> - Best-first search
> - In-Class

#### Problem -Solving Process for agent

- Goal Formulation
- Problem Formulation
- Search
- Execution* (often skipped)

#### Search Problem definitions

- State Space
  - Set of all possible states
- Initial state
  - State that the agent starts in
- Goal test
  - Goals for the agent, *Is-Goal* function

- Actions
  - Actions available to the agent, *Actions(s)* function
  - Input: State
  - Return: Collection of actions
- Transition Model
  - Description of what actions do, *Result(s, a)* function
  - Input: State, Action
  - Return; State
- Action-Cost Function
  - Numeric cost of actions, *Action-Cost(s, a, s')* function
  - Input: State, Action, State
  - Return: Numeric Cost
- Path
  - Sequence of actions
- Solution
  - Path from initial state to a goal state
- Optimal Solution
  - Solution with lowest cost for all actions in its path

#### Example Search Problem: Sliding Tile Puzzle

![截屏2023-01-30 16.22.50](/Users/huangjiatan/Library/Application Support/typora-user-images/截屏2023-01-30 16.22.50.png)

- 第一步把2/5/6/3向中间移动
- State size 9!
- *node*.STATE: the state to which the node corresponds;
- *node*.PARENT: the node in the tree that generated this node;
- *node*.ACTION: the action that was applied to the parent’s state to generate this node; 
- *node*.PATH-COST: the total cost of the path from the initial state to this node. In mathematical formulas, we use as a synonym for PATH-COST.

#### 8-Puzzle

- State
  - 2D-Array of int (-1 for blank)
  - 1D-Array of int (-1 or 0 for blank)
    - less memory
    - User friendly
    - Actions calculation easier
  - 9 Char string(space for blank)
  - in textbook use 9 tuple (less memory, efficient)
- Action
  - "R", "L", "U", "D"
- Node Object for Search Problems
  - State
  - Parent Node
  - Action: Action taken to get here
  - Path-Cost: Cost of path from initial state

#### Best-First Search: A Framework for multiple search functions

```python
def BestFirstSearch(problem, f):
  #initialize node, frontier, reached
  node = Node(state = problem.initial)
  frontier = PriorityQueue(order = f)#Queue()
  frontier.add(node)
  reached = {}
  reached[problem.initial] = node
  
  while not frontier.is_empty():
    node = frontier.pop()
    if problem.is_goal(node.state):
      return node
    for child in expand(problem, node):
      s = child.state
      if s not in reached or child.path_cost < reached[s].path_cost:
        reached[s] = child
        frontier.add(child)
  return failure
  
```

![Screen Shot 2023-02-27 at 14.22.19](/Users/huangjiatan/Library/Application Support/typora-user-images/Screen Shot 2023-02-27 at 14.22.19.png)

![IMG_7CA12069027E-1](/Users/huangjiatan/Downloads/IMG_7CA12069027E-1.jpeg)

## 2/1/2023 Lecture 5

> Writing #1
>
> In-class 2
>
> Best-first search
>
> - Example
>
> Algorithm Analysis
>
> - BFS, DFS, DLS, IDDFS

#### State Space

Tree-like

![IMG_60D560A6BFBE-1](/Users/huangjiatan/Downloads/IMG_60D560A6BFBE-1.jpeg)

#### Metrics for Search Algorithms

- Completeness
  - Guaranteed to find a solution if there is one
  - Report failure if not
- Cost Optimality
  - Does find optimal solution?
- Time Complexity
  - Big-O analysis for Time Complexity
- Space Complexity
  - Big-O analysis for Space Complexity

#### Best-First to Breadth-First

![Screen Shot 2023-02-27 at 14.28.20](/Users/huangjiatan/Library/Application Support/typora-user-images/Screen Shot 2023-02-27 at 14.28.20.png)

1. PriorityQueue(order = f) -> Queue()
2. First goal is the optimal one (don't need check path_cost)
3. Check is-goal inside the loop

- Analysis
  - Path cost are equal
  - There is a solution
  - Complexity Analysis
    - $1+b+b^2+...+b^d$
  - b: Branching factor
    - \# of children each node has (on average)
  - d: Depth of solution
  - m: Max depth of state space

|                   | BFS      | DFS (winner)    | DLS          | IDDFS (l start from 0) |
| ----------------- | -------- | --------------- | ------------ | ---------------------- |
| Completeness      | Yes*     | Yes*(no loops)  | Yes(if d<=l) | Yes*                   |
| Cost Optimality   | Yes**    | No              | No           | Yes**                  |
| Time              | $O(b^d)$ | worst: $O(b^m)$ | $O(b^l)$     | $O(b^d)$               |
| Space (Important) | $O(b^d)$ | $O(m*b)$        | $O(l*b)$     | $O(d*b)$               |

\* If b is finite, and state space either has a solution or is finite

** if action costs are equal

#### Best-First to Depth-First

![Screen Shot 2023-02-27 at 14.31.54](/Users/huangjiatan/Library/Application Support/typora-user-images/Screen Shot 2023-02-27 at 14.31.54.png)

1. PriorityQueue(order = f) -> Stack()
2. delete all about **reached**
3. Delete s = child.state and if s not in...

#### Depth-First Search Variants

- Depth-Limited Search
  - DFS but limit our search depth to a particular limit l
- Iterative-Deepening Depth-First Search
  - DFS but increase our search limit l by 1 until we find solution



## 2/6/2023 Lecture 6

> Uninformed Search
>
> - Review
>
> - Uniform-cost Search
>
> - Example
> - Analysis
>
> Informed Search
>
> - Greedy Best-Search
>
> - A*
>
> Inclass exercise

#### Uninformed Search

- because we don't know how close any given state is to a goal
- Uniform Cost
  - Although we don't know about distance from a node to a solution, we do know about distance from initial state to any given node.
  - Uniform Cost search orders nodes based on path cost
  - Assumption: All action costs are larger than some positive value $\epsilon$

![Screen Shot 2023-02-27 at 15.13.12](/Users/huangjiatan/Library/Application Support/typora-user-images/Screen Shot 2023-02-27 at 15.13.12.png)

#### expand() function

expand(problem, node)

![Screen Shot 2023-02-27 at 15.13.51](/Users/huangjiatan/Library/Application Support/typora-user-images/Screen Shot 2023-02-27 at 15.13.51.png)

#### Uniform-Cost Search

- Explore lowest path cost from initial state first
- Initial state defined to have path cost 0
- Reminder: All action costs are at greater than or equal to $\epsilon$ (some positive number)

|              | UCS                                      |
| ------------ | ---------------------------------------- |
| Complete     | Yes* (no loop)                           |
| Cost-Optimal | Yes                                      |
| Time         | $O(b^{1+ \lfloor c^*/\epsilon \rfloor})$ |
| Space        | $O(b^{1+ \lfloor c^*/\epsilon \rfloor})$ |

\* Action cost > $\epsilon$, $\epsilon$ is positive.

![Screen Shot 2023-02-28 at 19.13.50](/Users/huangjiatan/Library/Application Support/typora-user-images/Screen Shot 2023-02-28 at 19.13.50.png)

![IMG_DF1911FFD964-1](/Users/huangjiatan/Downloads/IMG_DF1911FFD964-1.jpeg)

#### Informed Search

- "Informed" because we do have an estimate of a node's distance to goal
- we have some function *h(n)* that takes a node as input and returns a numeric value that is an **estimate**
  - Heuristic: estimate of distance to goal
- Can use *h(n)* to order nodes
- Greedy Best-First: *f(n) = h(n)*
- A* Search: *f(n) = g(n) + h(n)*

| n    | h(n)  | n    | h(n)                        |
| ---- | ----- | ---- | --------------------------- |
| A    | 10    | H    | 2                           |
| B    | 6     | I    | 2                           |
| C    | 7 (5) | J    | 3                           |
| D    | 3     | K    | 2 (0) <- K is also solution |
| E    | 4     | L    | 0 <-L is the goal           |
| F    | 2     | M    | 1                           |
| G    | 1 (3) | N    | 3                           |
|      |       | O    | 2                           |

Frontier: (10, A)

Frontier: (6, B), (7, C)

Frontier: (3, D), (4, E), (7, C)

Frontier: (2, H), (2, I), (4, E), (7, C)



Frontier: (10, A)

Frontier: (5, C), (6, B)

Frontier: (7, F), (3, G), (6, B)

Frontier: (0, L), (1, M), (3, G), (6, B)



#### A* search

- Use h(n), a heuristic and g(n), path cost

- set Best-First search **f(n) = g(n) + h(n)**



## 2/8/2023 Lecture 7

> In class exercise
>
> Informed Search
>
> - A*
> - Examples
> - Analysis
>
> Heuristic Function

#### Q2 Simple state space

Suppose that we have a small state space as follows: The node with the initial state has 4 successors/children, and each of those have 3 successors/children. That is the entire state space. Assume that the solution state is the last one explored.

- What will the maximum length of the frontier be during a Breadth-First Search?
  - FGHIJKLMNOPQ
  - 12
- What will the maximum length of the frontier be during a Depth-First Search?
  - FGHCDE
  - 6

#### Informed Search Example

A*

Frontier: (10, A)

Frontier: (8, B), (10, C)

Frontier: (10, C), (11, D), (12, E)

Frontier: (10, F), (10, G), (11, D), (12, E)

Frontier: (10, G), (11, D), **(11, L)**, (12, E), (15, M)

Frontier:  (11, D), (11, L), (12, E), (12, N), (15, M), (16, O) //keep to check if has other shorter path

Frontier:  (11, I), (11, L), (12, E), (12, N), (13, H), (15, M), (16, O)

#### Route-Finding Example: Crookston to Rochester

in textbook

$h_{SLD}(n)$

Frontier: (350, C)

Frontier: (390, C-Mo), (400, C-D)

Frontier: (400, C-D), (405, C-Mo-Mp), (580, C-Mo-D)

Frontier: (405, C-Mo-Mp), (450, C-D-S), (460, C-D-Mp) ...

Frontier: (407, C-Mo-Mp-S), (450, C-D-S), (460, C-D-Mp)

Frontier: (417, C-Mo-Mp-S-R) 

#### A* Search: Analysis

- Is A* complete? What conditions are necessary?

  - Action costs > $\epsilon$ (any small positive number)

  - Finite state space (有一个支路的cost是1/2, 1/4, 1/8 ...)
  - finally will find a solution

- Is A* cost-optimal? What conditions are necessary?

  - Depend on h(n)

  - Inportant characteristic for h(n): **Admissibility**

  - An **admissible** heuristic function never overestimates

  - Always think that we closer

  - **Is cost-optimal**

  - Let the optimal path have cost C*

    - g(n) = path cost init to n
    - g*(n) = optimal path cost from init to n
    - h(n) = Heuristic function
    - h*(n) = cost of optimal path from n to goal

  - When it runs, it will return a path with cost C*

  - proof by contradiction: Assume that when I run A*

  - n* is on the optimal path, but did not get expanded by A*

  - what can I say about n?

    - **f(n') > C\***

    - f(n') = g(n') + h(n') (by definition)

    - f(n') = g*(n') + h(n')

      ​      <= g\*(n') + h\*(n')

      ​        = C* (by definition of C\*, g\*, h\*)
      
    - f(n') <= C* (collapsing last three lines)
    
  - Contradiction!
  

![Screen Shot 2023-02-27 at 15.21.27](/Users/huangjiatan/Library/Application Support/typora-user-images/Screen Shot 2023-02-27 at 15.21.27.png)

## 2/13/2023 Lecture 8

> - Writing #2
> - A*
> - Heuristics
> - Another Puzzle

#### A* Search: Analysis

Time complexity & Space complexity

- both depend on how good *h(n)* is.
- $O((b^*)^d)$

#### Heuristic Function

- How come up with good
- h(n) -> know goal, n has a state, estimate n state to goal state
- To ensure cost-optimality, we want it to be admissible
- Devise an estimate that is:
  - Admissible (i.e. never overestimates)
  - Faster to calculate than just searching
- Two common ideas for 8-pizzle
  - Number of wrong tiles (do not count the blank) - h1
  - Distance of wrong tiles to correct locations - h2
    - Orthogonal distance (sum of horizontal and vertical)
    - H2(n) = 0 + 3 + 2 + 3 + 1 + 2 + 1 + 1 = 13
  - Admissible? yes
  - how much time does it take to calculate? - 3*3
  - is one better than the other? - second one contains first one and get more

#### Heuristic Functions in general

- Relaxed problems
- look roles
  - Pick up and swap (h1)
  - Slide tiles on top of other (h2)

#### A* Search Variants

Exponential growth issues

possible solutions:

- Keep status only in one of *reached* or *frontier* (never both)
- Remove states from *reached* if we can prove that they are no longer needed (trade-off of time for space)
- Keep reference counts for states, remove from *reached* table when there are no more ways to reach that state (consider a grid world)
- Beam search: Limit the size of the frontier to only k items (trade-off)
- Iterative-Deepening A*: Like IDDFS, but increase search size by amount needed to get to next f-value.

#### Search: Summary

- Problem definition is key!
- Uninformed search:
  - if path-cost equal, IDDFS
  - If not, Uniform Cost
- Informed search:
  - A*
- Heuristic functions are crucial

#### N-Queens Problem

- Possible Problem-solving approach
  - State: Board with 0 or more queens
  - Action: Add a queen that doesn't attack others
  - Goal state: N queens on board
  - Heuristic function?
- Another approch
  - State: Board with N queens
  - Action: Move a queen
  - Goal State: No queens attack each other

## 2/15/2023 Lecture 9

> - In class exercise
> - HW2
> - Search in Complex Env
> - Local Search
>   - Hill climbing
>
> 

#### In class exercise

1. UCS action cost equal ->BFS

2. A* -> h(n) = 0

​		A* -> f(n) = g(n) + h(n) -> UCS

3. Straight-line distance between cities times 1.1 is not admissible for very close route (maybe same to straight-line)

#### Search in Complex Environments

- path to solution doesn't matter
  - Both discrete and continuous
- Nondeterminstic environments
- Partially observable environments
- Environment in which the Result(s, a) function can only be known by actually taking action a

#### Local search algoritm

- do not track path or states that have been reached
- low memory usage
- not systematic, may not get to entire state space
- solve puzzles (N-Queens)
- Optimization problems (don't know there's solution) (VLSI layout)

#### Hill Climbing Steepest Ascent

Value(s) function

Returns numeric

"Goodness" of state s



Consider a state space

b = 2 and moves that we can redo



```python
def hill_climbing_stochastic(problem, value):
  current = problem.initial
  while True:
    neighbors = get_all_better_successors(current, value)
    if len(neighbors) == 0:
      return current
    current = weighted_random_choice(neighbors)

```

```python
def hill_climbing_fc(problem, value):
  current = problem.initial
  while True:
    neighbor = get_new_random_successor(current)
    while value(neighbor) < value(current):
      neighbor = get_new_random_successor(current)
      if neighbor == None:
        return current
    current = neighbor
```

```python
def hill_climbing_rr(problem, value):
  while True:
    problem.initial = random_state(problem)
    result = hill_climbing_xyz(problem, value)
    
    if problem.is_solution(result):
      return result
```

#### Simulated Annealing

```python
def simulated_annealing(problem, schedule, value):
  current = problem.initial
  for t in range(1, ∞):
    T = schedule(t)
    if T == 0:
      return current
    next = random_successor(current)
    delta_E = value(current) - value(next)
    if delta_E > 0:
      current = next
    elif random.random() < e^{delta_e/T}:
      current = next
```

- unpacking the line: random.random() < $e^{delta_E/T}$
- random.random() - random float from 0 to 1
- Delta_E = value(current) - value(next)
  - But if it was greater than 0, we took the if statement
- T is current temperature
  - As T decreases, probability decreases

## 2/20/2023 Lecture 10

> Local search
>
> - more variants
> - Evolutionary Algorithms
>
> Continuous Environments
>
> Gradient Descent
>
> In-Class

#### Local Search

- **Steepest-Ascent Hill Climbing**: Always move to "best" successor

- **First-Choice Hill Climbing**: random "better" successor

- **Simulated Annealing**: Usually "good", maybe "bad"

  as time goes on, smaller possibabily make "bad"

- **Random-restart**: pick one from above, repeatedly start at random initial

#### Local Beam Search

- Track more than one node
- track k nodes, generate all successors, then choose k best
- k is problem-dependent
- When choosing successors, choose wirh weighted probabilities according to goodness

#### Evolutionary Algorithm

- Evolve states to find better valued states
- Population of "individuals" (states) from which the "fittest" (highest valued) get to produce "offspring" (successor states, kind of) via recombination
- State representation is a string over a finite alphabet
- one variable is a mixing number $\rho$ that is the number of individuals required to produce offsprings (i.e. how many parents, usually 2)
- Select that many individuals from population, weighted by fitness
- apply recombination procedure (15 pairs)
- ![Screen Shot 2023-02-20 at 16.54.20](/Users/huangjiatan/Library/Application Support/typora-user-images/Screen Shot 2023-02-20 at 16.54.20.png)
- Population -> fitness -> selection(random index) -> mutation(for each randomly, mutate 1 index) -> after all recombination, keep k for population in next "generation"
- Not complete

#### Search in Continuous Environment: Gradient Desent

- Continuous environments: State is based on continuous valued, not discrete

- Example: Airport placement

  - Airports are at locations (x1, y1) and (x2, y2)

  - 4-dimensional state space : s = (x1, y1, x2,  y2)

  - objective function f(s) = f(x1, y1, x2, y2) = 

    first, set C1 as the set of cities whose closest airport is #1, and C2 similarly

    Then, f(x1, y1, x2, y2) = $\sum dist(c,(x_1, y_1))^2 + \sum dist(c,(x_2, y_2))^2$

## 2/27/2023 Lecture 12

> Exam WEO
>
> HW 2
>
> Gradient Desent
>
> Nondet Environment
>
> In-class exercise



#### Exam

- 75 minutes
- 1 page notes (double sided)
- Topics: ==CH 2, 3, 4.1, 4.2==
- Closed book
- Like homework questions
  - Ch2 environment, agents
  - Ch3 algorithms 
  - Local search
- 4-5 questions

#### HW 2

question 2

question 4 a b c d

#### Gradient Descent: Airport Location Example

- s = (x1, y1, x2, y2)

- 2 airport, 8 cities

- a1(x1, y1), a2(x2, y2)

- Objective function f(s) = f(x1, y1, x2, y2) =  $\sum dist(c,(x_1, y_1))^2 + \sum dist(c,(x_2, y_2))^2$ - minimize this function

- Doesn't matter each airport serves exactly 4 cities (1 and 7 is fine)

- From calculate: Take the derivative, see where that is 0

- with more than 2 variables, Use Gradient of f

  $\nabla f = (df/dx1, df/dy1, df/dx2, df/dy2)$

  the collections C1 and C2 change

- ignore anything not involving x1

  $df/dx1 = 2\sum (x_i - x_c)$

- Our simplifying assumption isn't globally correct, but is locally correct

  - find slope at a given point
  - move in that direction by some amount
  - recalculate
  - $x = x + \alpha \nabla f(x)$ ---- $\alpha$ is relate to a lot (time ..)
  - repeat until
    - random restart
    - store value that find
    - Simulate annealing

#### Searching in Nondeterministic Environments

- no longer assume that actions will always get us to a single state
- **belief state**: Set of states that we could be in
- Vacuum-World
  - Faulty vacuum
  - Suck 
    - 80% clean current square
    - 20% clean current but also move dirty to neighbor
- Change `Result(s, a)` function to `Results(s, a)` that returns a set of states
- Single solution path is no longer sufficient
- **Conditional Plan**: Specification of what to do depending on percepts as plan is executed

## 3/13/2023

> final project prosposal
>
> Apa writing #3
>
> Searching in complex env
>
> - Nondeterministic
> - Partiality obs
> - Online

#### Search in Complex Environment (previously)

- Continuous(gradient decent)
- Nondeterministic
- Partiality obs
- Online

#### Searching  in Nondeterministic Environment: And-Or Search

State space will be a tree with two types of nodes

OR Node:

AND Node: 

![Screen Shot 2023-03-13 at 16.17.57](/Users/huangjiatan/Library/Application Support/typora-user-images/Screen Shot 2023-03-13 at 16.17.57.png)

![Screen Shot 2023-03-13 at 16.20.27](/Users/huangjiatan/Library/Application Support/typora-user-images/Screen Shot 2023-03-13 at 16.20.27.png)

- steps through the tree, finite plan to have a solution

- this is a conditional plan

#### Search in Partially Observable Environments

- **Belief state**: Set of states that we could be in
- Add a new function: Percept(s) that returns the percept received by the agent in the given state
- Note: It is possible that **multiple states will return the same percept**, consider vacuum-world
- Transition Model between belief states is in three steps:
  - Prediction: Compute estimated belief states resulting from possible actions
    - est_b = Result(b, a)
  - possible percept calculation: Compute possible percept for each state in estimated belief state
    - {o: o = Percept(s) and s $\in$ est_b}
  - Update: Compute for each possible percept, the belief state that would result from that percept
    - $b_o$ = Update(est_b, o) = {s: o = Percept(s) and s $\in$ est_b}
- Actions:
- Percept:

#### Online Agents and Unknown Environments

- "Online": Processing input as it is received (rather than waiting for entire data set to become available)

- Classic example:

- Online Agents generally know:

  - Actions(s)
  - c(s, a, s'): cost function. Note that this cannot be used until we know s'
  - IsGoal(s)

- Importantly, the agent cannot determine the return value from Result(s, a) except by actually being in state s and taking action a.

- <img src="/Users/huangjiatan/Library/Application Support/typora-user-images/Screen Shot 2023-03-13 at 16.46.37.png" alt="Screen Shot 2023-03-13 at 16.46.37" style="zoom:50%;" />

  - Where we've been? - result

  - Where walls are? - result

  - keep track IS_GOAL

  - Most recent decision point - untried

  - How to back up? - unbacktracked

  - ```
    s' = problem.init
    while True:
      a = online-obs-agent(p, s')
      if a = stop:
         return
      s' = result(s', a)
    ```

  - | Untried        | Result              | Unbacktracked    |
    | -------------- | ------------------- | ---------------- |
    | (1, 1): [U, R] | ((1, 1), U): (1, 2) | (1, 2): [(1, 1)] |
    | (1, 2): [D]    |                     |                  |
    |                |                     |                  |

## 3/15/2022

> - Final Project info
> - HW 3
> - Multiagent Environment
>   - Definitions
>   - Example Game Tree
>   - Minimax Algorithm

#### Adversarial Search

Are two player games nondeterministic environments?

#### Adversarial Search - Two player games - Terminology (Re-) Definitions

- Zero-Sum or Constant-Sum: What is good for one player is just as bad for the other
- Perfect information: Fully Observable
- Position: State
- Game Tree: State Space
- Move: Action - However, some games (like chess) use term "Move" to mean 2 actions.
- Ply: Action
- $s_0$: Initial state
- Max, Min: Names of the players (generally, agent is Max and opponent is Min)

#### Adversarial Search - Two-player games - Function Definitions

- Two-Move(s): Returns the player whose turn it is in state s
- Actions(s): Returns a collection of legal actions in state s
- Result(s, a): Returns the result of taking action a in state s
- Is-Terminal(s): Returns true if state s is a terminal state (i.e. game is over)
- Utility(s, p): Returns a numeric value to player p when the game ends in state s
- $s_0$  Actions($s_0$) -> [$a_1$, $a_2$, $a_3$]

#### Adversarial Search - State Space for simple game

#### Adversarial Search - Minimax Algorithm

```python
def minimax(game, s):
  player = game.To-Move(s)
  value, move = max_value(game, state)
  return move

def max_value(game, s):
  if game.Is-Terminal(s): return game.Utility(player, s)
	val = -infinity
  for a in game.Actions(s):
    v2, a2 = min_value(game, game.Result(s, a))
    if v2 > val: val, move = v2, a2
  return val, move

def min_value(game, s):
  if game.Is-Terminal(s): return game.Utility(player, s)
	val = infinity
  for a in game.Actions(s):
    v2, a2 = min_value(game, game.Result(s, a))
    if v2 < val: val, move = v2, a2
  return val, move
```

depth first traverse

pruning: 找每一个支路中最小的 max(3,0,2)

#### Adversarial Search - Minimax isn't good enough



