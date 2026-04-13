# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

torch_datas      = collect_data_files('torch', include_py_files=True)
torchaudio_datas = collect_data_files('torchaudio', include_py_files=True)
mutagen_datas    = collect_data_files('mutagen')
yt_dlp_datas     = collect_data_files('yt_dlp')
torch_binaries   = collect_dynamic_libs('torch')

a = Analysis(
    ['Music_Suite.py'],
    pathex=[],
    binaries=torch_binaries,
    datas=torch_datas + torchaudio_datas + mutagen_datas + yt_dlp_datas,
    hiddenimports=(
        collect_submodules('torch') +
        collect_submodules('torchaudio') +
        collect_submodules('mutagen') +
        collect_submodules('yt_dlp') + [
            'torchaudio.functional',
            'torchaudio.transforms',
            'tkinter',
            'tkinter.ttk',
            'tkinter.filedialog',
            'tkinter.messagebox',
            'pathlib',
            'threading',
            'shutil',
            'collections',
        ]
    ),
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Music Suite',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.icns',                    # ← remove this line if you have no icon yet
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Music Suite',
)

app = BUNDLE(
    coll,
    name='Music Suite.app',
    icon='icon.icns',                    # ← remove this line if you have no icon yet
    bundle_identifier='com.galfried.musicsuite',
    info_plist={
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.13.0',
    },
)
