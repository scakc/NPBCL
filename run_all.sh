### !/bin/sh
rm -r cache
rm -r saves
### Discriminative Experiments
# not MNIST
echo "Training not MNIST"
python3 npbcl_not.py
echo "saves\ncache/not_mnist" | python3 save.py
cp all_masks.png cache/not_mnist
cp union_mask.png cache/not_mnist
rm -r saves
# Split MNIST
echo "Training split MNIST"
python3 npbcl_split.py
echo "saves\ncache/split_mnist" | python3 save.py
cp all_masks.png cache/split_mnist
cp union_mask.png cache/split_mnist
rm -r saves
# Fashion MNIST
echo "Training fashion MNIST"
python3 npbcl_fashion.py
echo "saves\ncache/fashion_mnist" | python3 save.py
cp all_masks.png cache/fashion_mnist
cp union_mask.png cache/fashion_mnist
rm -r saves
# Permuted MNIST
echo "Training permuted MNIST"
python3 npbcl_perm.py
echo "saves\ncache/permuted_mnist" | python3 save.py
cp all_masks.png cache/permuted_mnist
cp union_mask.png cache/permuted_mnist
rm -r saves
## Generative Experiments
# MNIST (one digit at a time)
echo "Training vae on MNIST"
python3 npbcl_vae_mnist.py
echo "mnist" | python3 gen_extra.py
echo "saves\ncache/vae_mnist" | python3 save.py
echo "Gens\ncache/vae_mnist" | python3 save.py
rm -r Gens
rm -r saves
# not MNIST (one character at a time)
echo "Training vae on not MNIST"
python3 npbcl_vae_notmnist.py
echo "notmnist" | python3 gen_extra.py
echo "saves\ncache/vae_notmnist" | python3 save.py
echo "Gens\ncache/vae_notmnist" | python3 save.py