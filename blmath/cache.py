import os
from baiji.pod import AssetCache, VersionedCache
from baiji.pod.config import Config

config = Config()
config.CACHE_DIR = os.path.expanduser('~/.bodylabs_static_cache')
config.IMMUTABLE_BUCKETS = ['bodylabs-versioned-assets', 'bodylabs-versioned-assets-tokyo']
config.DEFAULT_BUCKET = os.getenv('ASSET_BUCKET', 'bodylabs-assets')

sc = AssetCache(config)

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), '..', 'manifest.json')
BUCKET = os.getenv('VC_BUCKET', 'bodylabs-versioned-assets')

vc = VersionedCache(
    cache=sc,
    manifest_path=MANIFEST_PATH,
    bucket=BUCKET)
