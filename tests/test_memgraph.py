from tests.test_framework import MemoryGraphManager

mg = MemoryGraphManager(promotion_threshold=2) # 测试环境阈值设低一点：2次
mg.show_status()