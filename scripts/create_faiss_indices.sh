# pip install clip-retrieval
# clip-retrieval inference \
#     --input_dataset "dataset" \
#     --output_folder "embeddings" \
#     --input_format files \
#     --num_prepro_workers 12 \
#     --enable_image True \
#     --enable_text False \
#     --enable_metadata False \
#     --clip_model "ViT-L/14" \
#     --use_jit False
# 
# clip-retrieval index \
#         "embeddings" \
#         "index" \
#         --max_index_memory_usage=4G \
#         --current_memory_available=16G \
#         --nb_cores 12                                                                                                                                                                               
