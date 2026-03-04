"""
版本检查脚本

运行此脚本以验证所有依赖包的版本是否正确安装
"""

import sys

def check_versions():
    """检查所有关键依赖的版本"""
    print("=" * 60)
    print("依赖包版本检查")
    print("=" * 60)
    
    required_packages = {
        'django': '4.2.0',
        'langchain': '0.3.0',
        'langchain_openai': '0.2.0',
        'langchain_community': '0.3.0',
        'langchain_core': '0.3.0',
        'langchain_text_splitters': '0.3.0',
        'openai': '1.0.0',
        'faiss': None,  # faiss-cpu 没有版本要求
    }
    
    all_ok = True
    
    for package, min_version in required_packages.items():
        try:
            # 尝试导入包
            if package == 'faiss':
                import faiss
                version = faiss.__version__ if hasattr(faiss, '__version__') else 'Unknown'
            else:
                mod = __import__(package)
                version = mod.__version__ if hasattr(mod, '__version__') else 'Unknown'
            
            # 检查版本
            if min_version and version != 'Unknown':
                from packaging import version as pkg_version
                if pkg_version.parse(version) >= pkg_version.parse(min_version):
                    status = "✅"
                else:
                    status = "⚠️"
                    all_ok = False
            else:
                status = "✅"
            
            print(f"{status} {package:30s} {version:15s} (要求: >={min_version or 'Any'})")
            
        except ImportError:
            print(f"❌ {package:30s} {'未安装':15s}")
            all_ok = False
        except Exception as e:
            print(f"⚠️ {package:30s} 检查失败: {e}")
    
    print("=" * 60)
    
    if all_ok:
        print("✅ 所有依赖包版本正确！")
        return True
    else:
        print("❌ 部分依赖包需要更新或安装")
        print("\n请运行以下命令安装/更新依赖:")
        print("pip install -r requirements.txt --upgrade")
        return False


def check_imports():
    """检查关键导入是否正常"""
    print("\n" + "=" * 60)
    print("导入测试")
    print("=" * 60)
    
    imports_to_test = [
        ("langchain_openai", "OpenAIEmbeddings", "OpenAI 嵌入模型"),
        ("langchain_openai", "ChatOpenAI", "OpenAI 聊天模型"),
        ("langchain_community.vectorstores", "FAISS", "FAISS 向量存储"),
        ("langchain_text_splitters", "CharacterTextSplitter", "文本分割器"),
        ("langchain_core.documents", "Document", "文档类"),
        ("langchain_core.messages", "SystemMessage", "系统消息"),
        ("langchain_core.messages", "HumanMessage", "用户消息"),
        ("langchain_core.messages", "AIMessage", "AI消息"),
    ]
    
    all_ok = True
    
    for module_name, class_name, description in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {description:20s} - {module_name}.{class_name}")
        except ImportError as e:
            print(f"❌ {description:20s} - 导入失败: {e}")
            all_ok = False
        except AttributeError as e:
            print(f"❌ {description:20s} - 类不存在: {e}")
            all_ok = False
    
    print("=" * 60)
    
    if all_ok:
        print("✅ 所有导入测试通过！")
        return True
    else:
        print("❌ 部分导入失败，请检查 LangChain 版本")
        return False


def check_environment():
    """检查环境配置"""
    print("\n" + "=" * 60)
    print("环境配置检查")
    print("=" * 60)
    
    import os
    from pathlib import Path
    
    # 检查 .env 文件
    env_file = Path('.env')
    if env_file.exists():
        print("✅ .env 文件存在")
        
        # 检查 OPENAI_API_KEY
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            print(f"✅ OPENAI_API_KEY 已配置 (长度: {len(api_key)})")
        else:
            print("⚠️ OPENAI_API_KEY 未配置")
    else:
        print("⚠️ .env 文件不存在，请创建并配置 OPENAI_API_KEY")
    
    # 检查数据库
    db_file = Path('db.sqlite3')
    if db_file.exists():
        print(f"✅ 数据库文件存在 (大小: {db_file.stat().st_size / 1024:.2f} KB)")
    else:
        print("⚠️ 数据库文件不存在，请运行: python manage.py migrate")
    
    # 检查 FAISS 索引
    index_dir = Path('faiss_index')
    if index_dir.exists() and (index_dir / 'index.faiss').exists():
        print("✅ FAISS 索引已构建")
    else:
        print("⚠️ FAISS 索引未构建，请运行: python manage.py build_faiss_index")
    
    print("=" * 60)


def main():
    """主函数"""
    print("\n🔍 开始检查项目环境...\n")
    
    # 检查 Python 版本
    print(f"Python 版本: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 版本过低，需要 3.8 或更高版本")
        return False
    
    # 检查依赖版本
    versions_ok = check_versions()
    
    # 检查导入
    imports_ok = check_imports()
    
    # 检查环境配置
    check_environment()
    
    # 总结
    print("\n" + "=" * 60)
    print("检查总结")
    print("=" * 60)
    
    if versions_ok and imports_ok:
        print("✅ 环境配置正确，可以正常运行！")
        print("\n下一步:")
        print("1. 确保 .env 文件中配置了 OPENAI_API_KEY")
        print("2. 运行数据库迁移: python manage.py migrate")
        print("3. 构建 FAISS 索引: python manage.py build_faiss_index")
        print("4. 启动服务器: python manage.py runserver")
        return True
    else:
        print("❌ 环境配置有问题，请根据上述提示修复")
        print("\n建议:")
        print("1. 运行: pip install -r requirements.txt --upgrade")
        print("2. 重新运行此脚本验证")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ 检查被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

