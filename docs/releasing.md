# Releasing

Releases are driven by tags. Pushing `vX.Y.Z` runs the `Release` workflow:

1. **Build**: checks that the tag matches `version` in `pyproject.toml`, builds the sdist and wheel, and runs `twine check --strict`.
2. **Publish**: uploads to PyPI with one of two methods.
3. **Docker**: builds the image for `linux/amd64` and `linux/arm64` and pushes it to `ghcr.io/sadrmanesh/persian-rag` with the tags `X.Y.Z`, `X.Y`, and `latest`. It signs in with the workflow's `GITHUB_TOKEN`, so no extra secret is needed.

| Method | Setup |
| --- | --- |
| Trusted publishing (recommended) | Add a trusted publisher on PyPI for this repository, workflow `release.yml`, environment `pypi`. Then set the repository variable `PYPI_TRUSTED_PUBLISHING` to `true`. |
| API token | Add a `PYPI_API_TOKEN` repository secret. |

If neither is configured, the build still runs, the publish job prints a notice, and nothing is uploaded.

## Checklist

1. Update `version` in `pyproject.toml` and `src/persian_rag/version.py`.
2. Add a section to `CHANGELOG.md`.
3. Merge to `main` with green CI.
4. Tag and push: `git tag v0.4.0 && git push origin v0.4.0`.

The release workflow also copies the GHCR image, for every platform, to Docker Hub as `docker.io/<DOCKERHUB_USERNAME>/persian-rag` with the same `X.Y.Z`, `X.Y`, and `latest` tags. This job runs only when the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets are set, and skips with a notice otherwise.

## Mirrors

`.github/workflows/mirror.yml` pushes `main` and every tag to `gitlab.com/sadrmanesh/persian-rag`. It runs only when the `GITLAB_TOKEN` repository secret is set (a GitLab access token with the `write_repository` scope) and skips with a notice otherwise. Set the `GITLAB_NAMESPACE` repository variable to mirror to another GitLab namespace. GitHub stays the main repository for issues and pull requests.
