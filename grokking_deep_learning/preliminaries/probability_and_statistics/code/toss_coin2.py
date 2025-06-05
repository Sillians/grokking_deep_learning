# %matplotlib inline
import torch
from matplotlib import pyplot as plt
from torch.distributions.multinomial import Multinomial
from d2l import torch as d2l

fair_probs = torch.tensor([0.5, 0.5])
print(Multinomial(100, probs=fair_probs).sample())


"""
Each time you run this sampling process, you will receive a new random value that may
diﬀer from the previous outcome. Dividing by the number of tosses gives us the frequency
of each outcome in our data. Note that these frequencies, just like the probabilities that they
are intended to estimate, sum to 1.
"""

print("frequency: ", Multinomial(100, probs=fair_probs).sample() / 100)


# Let’s see what happens when we simulate 10,000 tosses.
counts = Multinomial(10000, probs=fair_probs).sample()
print("simulate 10000 counts: ", counts / 10000)



# Let’s get some more intuition by studying how our estimate evolves as we grow the number of tosses from 1 to 10,000.
counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
estimates = estimates.numpy()
print(estimates)

d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();





