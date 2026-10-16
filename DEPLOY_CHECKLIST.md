# 静态站点部署检查

## 构建与验证

- [ ] `python -m pip install -r requirements-build.txt`
- [ ] `python -m unittest discover -s tests -v`
- [ ] `python scripts/build_site.py`
- [ ] `python scripts/verify_site.py`
- [ ] `node --test tests/planner.test.mjs`
- [ ] 通过本地 HTTP 服务检查目标潜能 / 目标抽数、复刻开关及两池水位。
- [ ] 检查三档推荐、详情 PMF / CDF、区间查询、整数反查、武器与联合目标。
- [ ] 检查手机宽度下的表单、长标题和图表，无页面横向溢出。

只需部署 `site/`，其中应包含 `data/manifest.json`、163 个二进制数据文件、`vendor/` 图表库、HTML、CSS 和 JavaScript。生成目录由构建脚本创建，不提交到 Git。

## GitHub Pages

- [ ] 改动合入 `master`。
- [ ] Settings → Pages → Source 设为 GitHub Actions。
- [ ] `Test and deploy GitHub Pages` 的 build 与 deploy 均成功。
- [ ] 在仓库子路径地址验证数据与 worker 正常加载。

工作流先校验模型和数据，再发布。无需额外后端、账户密钥或 Streamlit Cloud。

## 更新规则

修改概率规则后更新 `MODEL_VERSION`，重新构建并运行校验。静态数据记录模型源文件哈希；源与数据不一致时校验失败。旧联合缓存通过版本化键隔离，不能用于新版结果。

原有 Streamlit 若继续使用，应另外安装 `requirements.txt` 并运行 `python precompute_joint_results.py`，生成修正后的联合缓存。
