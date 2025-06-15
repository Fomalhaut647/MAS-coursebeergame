import matplotlib.font_manager as fm

# 1. 列出所有 Matplotlib 可用的字体名称
# 这一步会刷新 Matplotlib 的字体列表，但不会删除缓存文件
print("=== Matplotlib 可识别的字体（包含子集）：===")
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf') # 扫描系统所有ttf字体
font_names_set = set()
for font_path in font_list:
    try:
        prop = fm.FontProperties(fname=font_path)
        font_names_set.add(prop.get_name())
    except Exception as e:
        # 忽略无法加载的字体
        pass

# 打印所有 Matplotlib 能够识别的字体名称，查找包含“PingFang”的名称
for font_name in sorted(list(font_names_set)):
    if 'PingFang' in font_name or '苹方' in font_name:
        print(font_name)