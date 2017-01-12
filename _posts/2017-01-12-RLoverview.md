---
layout: post
title: An Informal Survey of Current Challenges in Reinforcement Learning
---

$$ a + b = \epsilon $$

\section{Introduction: Challenges in Reinforcement Learning}
\paragraph{State-of-the-art}
Reinforcement learning is probably the most succesful learning paradigm for intelligent autonomous behaviour \cite{mnih2015human,silver2016mastering}. The current state-of-the-art is provided by (asynchronous) policy search methods \cite{mnih2016asynchronous,schulman2015high}. Although they show promising results in complicated scenario's, there remain several limitations. The first one is practical, as getting algorithms to work requires quite some experience and hand-tuning. However, we will here focus on more fundamental RL challenges, which may help overcome these 'domain-knowledge' challenges as well.   

\paragraph{Challenges}
For long, the major major limiting factor for scaling reinforcement learning algorithms was representation learning. However, recent progress in deep learning has created important progression in this direction. Thereby, many of the traditional RL challenges ´return' to the research agenda, but can be tested in more complicated scenario´s. We will shortly identify a list of major challenges, which is partially based on a talk by Pieter Abbeel at the International Joint Conference on Artificial Intelligence (IJCAI) 2016: 

\begin{enumerate}
\item {\bf Learning algorithms}: Much of the (initial) work on RL focusses on different learning algorithm styles, e.g. value-based methods versus policy search, bootstrapping or Monte Carlo roll-outs, etc. The current state-of-the-art results are provided by Actor-Critic methods \cite{mnih2016asynchronous}, which can also be categorized as policy search algorithms with a value function baseline \cite{schulman2015high}. 
\item {\bf Model-based learning}: Stable estimation of (stochastic) transition models has been a challenge in RL and optimal control communities for decades. We will investigate the benefits and challenges of model-based RL in detail in RQ 1 and 2.  
\item {\bf Exploration}: Many state-of-the-art learning algorithms still base itself on ($\epsilon$-greedy) exploration. A good example is the inability of current algorithms to solve the Atari game Montezuma´s Revenge, which has very long exploration sequences before a sparse reward occurs. There is extensive theoretical work on model-based exploration, but these methods tend to break down in more complex scenario´s. There is also much research on intrinsic motivation, but these usually require model estimates as well. In general, it seems that stable model estimation remains a limiting factor for targeted exploration. See chapter RQ3. 
\item {\bf Memory}: Another yet unsolved challenge in RL is violation of the Markov assumption, i.e. the need for long-term memory. See chapter RQ4. 
\item {\bf Learning from humans}: We identify three important directions in this class. The first is learning from demonstration (LfD). The cardinal example is Inverse Reinforcement Learning, which aims to estimate the human reward function from demonstration. The second group involves learning from human feedback, which studies how to translate the feedback into the policy. Finally, a third category is human-robot-interaction, including social signals like emotions. (cf. separate document).  
\item {\bf Hierarchy \& Temporal abstraction} \cite{barto2003recent}: Identifying hierarchy in RL tasks is the control variant of representation learning, i.e. identifying (multiple) layers of abstraction in data/tasks \cite{schmidhuber2015deep}. Probably the best known framework are MAXQ \citep{dietterich2000hierarchical} and options \cite{sutton1999between}. Both describe methods for solving the hierarchiecal problem given the decompositions is available. An even bigger challenge is the automatic sub-goal {\it discovery}. There is other work that addresses automatic option discovery, e.g. \citep{machado2016learning}, graph partitioning-based \cite{csimcsek2005identifying}, universal option models \citep{szepesvari2014universal} and linear options \cite{sorg2010linear}. Others have jointly modelled states and goals \cite{schaul2015universal}.  
\item {\bf Transfer}: See final section in next chapter.
\end{enumerate}

\paragraph{Robotics experiments}
The transfer of simulation results to real robotic experiments is non-trivial. Real robots can only learn in real wall-clock time, and are moreover vulnerable to physical harm, both greatly limiting the amount of acquirable experience. However, there are a few approaches that leverage end-to-end training in robotic scenario's. The prime example is Guided Policy Search \citep{levine2016end}, where trajectory optimization techniques are used to generated informative demonstration trajectories, to which an RL-like optimization is fitted. Other work uses results from transfer learning to accommodate the transition \citep{rusu2016sim}. 
