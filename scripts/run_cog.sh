# text prompt
# cog predict \
#     -i prompts="oil painting of pembroke welshi corgi" \
#     -i negative_prompt="watermark" \
#     -i use_database=True \
#     -i database_name="openimages" \
#     -i scale=5.0 \
#     -i seed=42 \
#     -o output.json

# image prompt
# cog predict \
#     -i image_prompt=@test.png \
#     -i scale=5.0 \
#     -i seed=42 \
#     -o output.json