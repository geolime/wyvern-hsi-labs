import rioxarray as rxr

URL = "https://wyvern-prod-public-open-data-program.s3.ca-central-1.amazonaws.com/wyvern_dragonette-001_20250724T071146_fbaa00bd/wyvern_dragonette-001_20250724T071146_fbaa00bd.tiff"

da = rxr.open_rasterio(
    URL,
    masked=True,
    chunks={"x": 1024, "y": 1024},
)

print(da)
print("dims:", da.dims)
print("shape:", da.shape)

# Force a tiny read (proves it actually works end-to-end)
sample = da.isel(band=0, y=slice(0, 10), x=slice(0, 10)).values
print("sample shape:", sample.shape)
print("ok")
