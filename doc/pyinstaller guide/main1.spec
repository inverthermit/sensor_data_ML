# -*- mode: python -*-

block_cipher = None

project_dir = 'D:\\Outotec_Work_Files\\project\\lanceAI\\sensor_data_ML\\'

added_files = [
         (project_dir +'config.json', '.'),
	(project_dir +'data\\trials\\*.json','.//data//trials'),
(project_dir +'script\\feature', 'feature' ),
(project_dir +'script\\util', 'util' )
         ]

a = Analysis(['main.py'],
             pathex=[project_dir +'script\\process'],
             binaries=[],
             datas= added_files,
             hiddenimports=['pandas','sklearn','sklearn.neighbors','sklearn.neighbors.typedefs','sklearn.neighbors.quad_tree','sklearn.tree._utils'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='main',
          debug=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='main')
