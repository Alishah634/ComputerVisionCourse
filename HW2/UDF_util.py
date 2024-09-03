# def parse_pqrs(file_path, num_images=4):
#     points_dict = {}
#     pattern = re.compile(r'img(\d) (\w):\s*P\((\d+),(\d+)\)\s*,?\s*Q\((\d+),(\d+)\)\s*,?\s*R\((\d+),(\d+)\)\s*,?\s*S\((\d+),(\d+)\)')
#     with open(file_path, 'r') as file:
#         # content = file.read()
#         l = 0
#         for line in file.readlines():
#             print(line)
#             where_is_img = line.find("img")
#             img_number = where_is_img+1
#             print(img_number)
    
#             # points_dict[img_key] = {'P': None, 'Q': None, 'R': None, 'S': None}
#             # print(f"Error: Points not found for {img_key}")
#     return points_dict
