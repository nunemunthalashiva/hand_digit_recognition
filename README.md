#### Here we will be building very basic and simple neural network . We will be training on MNIST dataset and test the results.

We had 3 files here 
<ul>
      <li> load_dataset : which had a function "load_dataset" which loads our MNIST dataset .</li>
      <li> implementation.py : Here we had the functions "SGD" (which is essentially mini batch)</li>
                           &ensp;      We also had "update_mini_batch" which updates parameters batch wise<br>
                           &ensp;   "feed_forward" it just return the prediction value based on weights and biases<br>
                           &ensp;  "backpropogate" which essentially does backpropogation .<br>
                           &ensp;  and others are small helper functions.
                             
</ul>

### Backpropagation algorithm
<ul>
  <li>The backpropagation algorithm provide us with a way of computing the gradient of the cost function by performing the following operations</li>
  <li>Feed forward :for each l=2,3...L we compute z<sup>l</sup> = (w<sup>l</sup>)(a<sup>(l-1)</sup>) + b<sup>l</sup>   and a<sup>l</sup> = σ(z<sup>l</sup>)
  <li>Output error: 𝛿<sup>l</sup> = ∇<sub>a</sub>(cost_function) * σ<sup>1</sup>(z<sup>l</sup>)  (Note: "*" here is dot product)</li> 
  <li>Backpropagation error : 𝛿<sup>l</sup> = (w<sup>l+1</sup>)<sup> T</sup> 𝛿<sup>l+1</sup> *σ <sup>1</sup>(z<sup>l</sup>)</li>
</ul>

### Neural Network's Output
<ul>
  <li>Our neural network has only one hidden layer having 30 neurons.(its a hyperparameter we got to know if this is 30 its showing minimum error rate .)</li>
  <li>The final accuracy we are getting is around 95%.</li>
 </ul>

