---
layout: post
title: An Informal Survey of Challenges in Reinforcement Learning (With References)
---

Reading up on a new research topic is tricky, and takes about the first year of your PhD. For reinforcement learning (RL), you have a good start with the Sutton and Barto book. Then, you could also try the more recent book by Wiering and van Otterloo. However, progress in (deep) RL is so fast, that you really aren't done yet. What I found most difficult in the beginning was to figure out *what* are the remaining challenges, and what people are working on. To save you the hassle, I will try to informally summarize. 

**Challenges**

For long, the major major limiting factor for scaling reinforcement learning algorithms was representation learning. However, recent progress in deep learning has created important progression in this direction. Thereby, many of the traditional RL challenges ´return' to the research agenda, but can be tested in more complicated scenario´s. This list is partially based on a talk by Pieter Abbeel at the International Joint Conference on Artificial Intelligence (IJCAI) 2016: 

1. *Learning algorithms*: The bulk of RL research has focussed on different learning algorithm's, e.g. value-based methods versus policy search, bootstrapping or Monte Carlo roll-outs, etc. The current state-of-the-art results are provided by Actor-Critic methods {% mnih2016asynchronous % %}, which can also be categorized as policy search algorithms with a value function baseline {% schulman2015high %}. 
2. *Model-based learning*: Stable estimation of (stochastic) transition models has been a challenge in RL and optimal control communities for decades.   
3. *Exploration*: Many state-of-the-art learning algorithms still base itself on ($\epsilon$-greedy) exploration. A good example is the inability of current algorithms to solve the Atari game Montezuma´s Revenge, which has very long exploration sequences before a sparse reward occurs.  
4. *Memory*: Another yet unsolved challenge in RL is violation of the Markov assumption, i.e. the need for long-term memory.
5. *Transfer*: We would ideally find a RL algorithm that is able to take it's knowledge to the next task and built from the previous task (especially if they are related). 
6. *Hierarchy & Temporal abstraction*: Hierarchy is very particular to the temporal aspect of RL (compared to e.g. a non-temporal classification or regresssion task). Where deep architecture learn hierarchiecal representations within a picture, we would ideally do the same thing on a temporal scale. These subpatterns may then be re-used in new tasks (i.e., hierarchy is close to transfer, although it may also speed-up learning within the same task).   
7. *Learning from humans*: Finally, the feedback for RL settings may be provided in an interactive context. Human information is noisy, impatient, and crucially depends on reciprocal communication, all complicating the learning process. 

I think all of these have one underlying target: finding an optimal (or close) solution as fast as possible (data efficiency), over a lwide variety of tasks.

**Model-based reinforcement learning**

Succesful application of model identification dates back to the locally weighted regression techniques of {% atkeson1997locally %}. By using the linearization only locally, i.e. iteratively refitting it, we limit the hazard of linearization. Nevertheless, linearized dynamics pose obvious limitations in representational capacity. 

Non-linear model identification through machine learning is a much harder topic. Earlier work studied random forests {% hester2012learning %} or nearest-neighbours {% jong2007model %} to identify non-linear predictions. Recent work used deep convolutional neural network to predict in high-dimensional settings up to the pixel-level {% oh2015action %}. {% wahlstrom2015pixels %} also uses deep feedforward network to approximate model dynamics, after pretraining with deep auto-encoders. {% heess2015learning %} uses feedforward networks to predict the relative change in state. Value-iteration networks {% tamar2016value %} are another way to estimate transition models, by embedding it as convolutional operations in the neural network itself. {% watter2015embed %} does investigate deep generative models (VAE), but constrains the latent level dynamics to be linear again.

*Planning with models* Work on model-based acceleration originates in the work on Dyna-Q {% sutton1991dyna %}, which in between true sampling updates incorporated planning-like updates to the value function.    

On direction to solving the planning problem is through sampling. The potential of model-based acceleration with a perfect model was already illustrated by AlphaGo {% silver2016mastering %}, who gained important performance benefit from incorporating Monte Carlo Tree Search {% couetoux2011continuous %}. In case of a know, given model, MCTS planning was already shown to be able to generate informative training data {% guo2014deep %}. Early versions of this work are TD-planning {% silver2008sample %}. {% gu2016continuous %} uses models to generate roll-outs, by taking off-policy targets in the real-world and on-policy targets in synthetic roll-outs. However, they find local linear models are better in terms of learning efficiency, as their neural network requires too many samples to be approximated. 

Another direction of planning solutions given a model are based on differentiating the policy (policy gradients). For example, PILCO {% deisenroth2011pilco %} uses Gaussian processes for model estimation in combination with policy gradients. In the optimal control community, the most popular approach is iterative LQG (iLQG) {% todorov2005generalized %}, which uses locally optimal feedback controllers with linearized dynamics. The main benefit is that we can then analytically derive the optimal policy from the quadratic value functions. An extension of this method is Guided Policy Search {% levine2016end %} in case of unknown dynamics. {% heess2015learning %} developed rules for differentiating with respect to stochastic transition functions.

**Exploration**

Although reinforcement learning has shown success in recent high-dimensional applications, these methods generally still used `undirected' exploration {% thrun1992role %} strategies. Most publications use $\epsilon$-greedy {% mnih2015human %}, Boltzmann or Gaussian noise exploration. However, such heuristic approaches have random-walk exploration behaviour, which may show exponential regret even for small MDP's {% osband2014generalization %}. The recent deep RL successes battle this problem through extensive exploration, i.e. running their algorithm for multiple tens of millions of frames. This approach is clearly infeasible in real-world applications.

There is however extensive work on exploration. First, in smaller domains there are several formal guarantees {% brafman2002r %}, mostly based on optimism under uncertainty in combination with count-based exploration. In general, the derived guarantees only work for discrete action space, and tend to computationally break down in large state-space.

In larger state-space, we may discriminate count-based exploration and intrinsic motivation literature. In the count-based direction, there is less work available. One recent work explicitly uses a model to plan towards states with low similarity compared to recently visited frames {% oh2015action %}. Mention {% osband2016deep %}. 

Intrinsic motivation {% oudeyer2007intrinsic,schmidhuber2010formal %} typically adds some derivative of `novelty' to the extrinsic reward functions, making the agent act on aspects of the current model and environment interaction history. Examples are from {% houthooft2016curiosity %}, who use Bayesian Neural Networks, i.e. variational inference on the model parameters (compare to section on variational auto-encoders), to quantify the uncertainty in the model estimates, and use these to estimate 'surprise'. Other examples use intrinsic motivation based on mutual information {% mohamed2015variational %}, prediction error {% stadie2015incentivizing %}, or .... {% hester2015intrinsically %}. Recent work {% kulkarni2016hierarchical %} leverages intrinsic motivation in a context of hierarchical RL, showing promising exploration on the challenging Montezuma's revenge game.  
 
Finally, {% bellemare2016unifying %} recently showed that both count-based exploration and intrinsic motivation are unifyable in a single paradigm through `pseudo-counts'. Pseudo-counts are effectively density estimates of visiting frequency over state-space. 

**Memory**

The Markov assumption is one of the fundamental assumptions underlying the MDP formulation. When an informed decision in a particular state actually require information that was only observable in a previous state, an ordinary RL algorithm will fail. We then enter the domain of partially-observable markov decision processes (POMPD's). Work on POMDP in tractable domains is reviewed in {% spaan2012partially %}. 

The dominant approach for solving memory has been through RNN's {% wierstra2010recurrent %}, or a variant called long-short term memory (LSTM). RNN's have been applied to robotics {% bakker2003robot %}, and deep RL settings {% hausknecht2015deep %}. Other examples append open continuous memory states to the state-space {% zhang2016learning %}.  

*Related work in deep learning: Attention* Another recent advancement in (deep) machine learning are succesful implementations of `attention'. Attention models {% mnih2014recurrent %} are recurrent neural networks (RNN), where the network iteratively processes an input to provide a prediction. In each iteration, the network also provides a new attention location, i.e., where to focus next. In general, we need to discriminate between {\it hard %} and {\it soft %} attention. As deciding where to attend is a stochastic operation, through which we want to back-propagate, we have two options. In the hard attention scheme, we sample a single decision, and use the REINFORCE trick to propagate the gradients {% mnih2014recurrent %}. In the soft attention scheme, we take a weighted average of all attention targets (for example through a softmax function).  

Another application of attention mechanisms is in Neural Turing Machines (NTM) {% graves2014neural %}. Here, instead of attending over the input data, the network learns to interact with an external memory, which functions much like a computer's RAM. The network learns to interact with this memory through read and write operations. Deciding where to read and write is a similar attention problem, where the mechanism may again be soft {% graves2014neural %} or hard {% zaremba2015reinforcement %}. A variant of NTM's are Memory Networks {% sukhbaatar2015end %}. 

There are recent applications of the Neural Turing Machine to reinforcement learning tasks. {% oh2016control %} uses both feedforward and recurrent architectures based on Memory Networks {% sukhbaatar2015end %}. The read and write operations are based on content-based retrieval, i.e., at each timestep the network generates a key, which is then compared to each entrie of the memory. Recent work by {% graves2016hybrid %} tested a NTM variant known as differential neural computer, which has read operations based on content and a temporal linkage system, while the write operations are based on content and usage.

**Transfer**

Transfer is the challenge of re-using knowledge from earlier (source) domains in a new (target) domain. MDP tasks can be identified based on their state/action space, environment dynamics and reward function. We can categorize transfer instances as follows:
1. Same state/action space, same dynamics, different reward function.
2. Same state/action space, different dynamics, same/different reward.
3. Different state/action space (implying changes in the other two as well.).

There are a variety of solutions to transfer in RL. Some examples of transfer involve:
1. Representation transfer: We can re-use the initial layers of our networks {% parisotto2015actor %}, or add lateral connections from each layer to a new output network {% rusu2016progressive %}. 
2. Hierarchy: See next topic.  is a long-standing RL challenge, which is closely intertwined with transfer. There is a variety of work on automatically inferring {\it options %} in a MDP task (e.g. {% machado2016learning %}). Arguably, hierarchy is a form of temporal abstraction/representation learning. 

*Related work in deep learning: One-shot generalization* Finally, the models now also extend their reach to the domain of 'one-shot generalization' {% rezende2016one %}. This domain is driven by the capabilities of humans to generalize from single examples, which are much different from the abilities of standard function approximators {% lake2016building %}. One-shot learning is effectively a special instance of transfer learning, i.e. where the source and target task are similar, but the number of new examples is very small. It turns out that Memory Networks {% santoro2016one %} or NTM's can `meta-learn' structure in a task, which allow it to store and retrieve relevant examples after a single presentation. 

**Hierarchy**

{% barto2003recent %} Identifying hierarchy in RL tasks is the control variant of representation learning, i.e. identifying (multiple) layers of abstraction in data/tasks {% schmidhuber2015deep %}. Probably the best known framework are MAXQ {% dietterich2000hierarchical %} and options {% sutton1999between %}. Both describe methods for solving the hierarchiecal problem given the decompositions is available. An even bigger challenge is the automatic sub-goal {\it discovery %}. There is other work that addresses automatic option discovery, e.g. {% machado2016learning %}, graph partitioning-based {% csimcsek2005identifying %}, universal option models {% szepesvari2014universal %} and linear options {% sorg2010linear %}. Others have jointly modelled states and goals {% schaul2015universal %}.

**Learning from human feedback**

I will not go into detail here. We can in general identify two ways in which human feedback may influence a MDP learning process: Learning from Demonstration (LfD) and Learning from Experience (LfE) (or Learning from Critique) {% argall2009survey %}. The difference between both centers around who is executing the task: teacher respectively learner. Note that there exists a term Reinforcement Learning from Demonstration (RLfD), which is essentially the combination of the two: the process is initialized by from demonstrations, and further fine-tuned through trial-and-error learning. {% schaal1997learning %}

One of the most explored concepts in this area is Inverse Reinforcement Learning, which tries to identify a reward function from observed expert demonstrations. 

References
----------

{% bibliography --cited % %}