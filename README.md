# FT-Transformer
原作者的colab範例程式碼<a href="https://colab.research.google.com/github/Yura52/rtdl/blob/main/examples/rtdl.ipynb#scrollTo=eb3Y6bnuVNpG" rel="nofollow"><img src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;"></a>  
原作者的github連結: <a> https://github.com/Yura52/tabular-dl-revisiting-models </a>  

# 使用說明
1. 所有可用參數及說明皆在main.py裡，可以直接去裡面查看與設定，下方為使用範例
<pre>
cd fttransformer
python main.py --data_dir ${data_dir} --epoch 10 --lds --reweight "sqrt_inv" --fds --fds_kernel 'gaussian' --fds_ks 5 --fds_sigma 1 --start_update 0 --start_smooth 1 --bucket_num 15 --bucket_start 3 --fds_mmt 0.9
</pre>
2. 程式碼裡內建的模型是FT-Transformer，並導入LDS與FDS模塊
3. 若需要應用於新的資料集，需要自行新增相關的資料處理於main.py，並將main.py第72-102行改成您的Data Preprocess
