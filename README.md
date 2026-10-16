# 终末地 · 寻访规划

可部署到 GitHub Pages 的静态抽卡计算器，同时保留原有 Python CLI 和 Streamlit 入口。

- **目标潜能**：计算 50% / 90% / 99% 达成概率下的最少总抽数。
- **目标抽数**：计算固定预算在三档概率要求下至少能达到的潜能。
- **两池分配**：可选择首发 + 同角色首次复刻，每档独立优化本池、复刻抽数。
- **详细计算**：干员、武器、联合目标；均值、标准差、区间概率、整数概率反查、PMF / CDF 图与完整明细。

## 本地构建与预览

构建使用 Python 3.11+ 和 NumPy，JavaScript 测试使用 Node.js 24。网页运行时无需 Python 服务。

```powershell
python -m pip install -r requirements-build.txt
python scripts/build_site.py
python -m http.server 4173 --bind 127.0.0.1 --directory site
```

打开 <http://127.0.0.1:4173/>。请通过 HTTP 预览；直接双击 HTML 无法加载概率数据和模块。

构建脚本生成 `site/data/`，并从 Plotly 官方 CDN 下载固定版本的图表运行库到 `site/vendor/`。这两个目录不提交到 Git，由 GitHub Actions 重新构建。运行期间的脚本、图表和数据都由本站提供，按需加载选中的水位数据。

## 发布到 GitHub Pages

1. 将改动合入 `master`。
2. 仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。
3. 在 **Actions → Test and deploy GitHub Pages** 查看构建，或使用 **Run workflow** 手动触发。

工作流运行 Python 测试、构建、完整数据校验和 JavaScript 测试后上传 `site/`。只有 `master` 上的推送或手动运行会部署，PR 只构建验证。

发布后的默认地址为 <https://bcszsz.github.io/endfield-draw-calculator/>。所有资源使用相对路径，支持仓库子路径。

## 验证

```powershell
python -m unittest discover -s tests -v
python scripts/build_site.py
python scripts/verify_site.py
node --test tests/planner.test.mjs
```

数据校验覆盖所有 160 份干员分布、武器分布、25,920 个联合目标场景、概率质量与单调性。Python 独立枚举产生两池规划答案，与 JavaScript 规划及固定执行方案的累计概率交叉核对。源模型哈希变化后，旧数据会被校验拒绝，需重新构建。

## 计算口径与范围

- 默认尚未持有目标角色。本体为 0 潜，需要 1 份；满潜为 5 潜，需要 6 份。
- 主规划总预算为 0–1200 次常规寻访；六星初始水位为 0–79，可分别设置本池和复刻。
- 本池与首次复刻的累计奖励、120 保底进度均从零开始，两个卡池的六星水位独立。
- 六星基础概率 0.8%，UP 占 50%。连续 65 抽未出六星后，第 66 抽开始每抽提升 5 个百分点，第 80 抽保证六星。
- 首发第 30 抽送加急十连；首次复刻第 30 / 60 / 90 抽各送一次。加急结果按基础概率结算，不改变常规六星水位或 120 保底状态。
- 每个池子的前 120 次常规寻访保证一次 UP。免费十连出 UP 不消耗该保证；120 保底是该次寻访的结果，不能在普通结果之外再补一抽。
- 每累计 240 次常规寻访直接赠送信物，不影响保底、不产生武库配额。
- 同名重构寻访的赠送、120 / 240 进度跨期继承，不能在每次后续复刻中重置。本版规划范围为首发与首次复刻，不计算途中其他卡池的收益。
- 分配为事先确定的两池预算上限；达到目标可以提前停止。同潜能时优先提高成功概率，完全等价时优先本池。
- 推荐方案详情固定“本池最多抽 N 次，之后在复刻继续至达标”的执行方式，计算其完成抽数 PMF / CDF；不会把不同预算下的最优分配混成一条完成分布。
- 武器沿用原有按十连结算的模型：4% 六星、1% UP，第四次十连六星保底、首次第八次十连 UP 保底，第 18 / 34 / 50 / … 次十连赠送 UP。忽略武库箱价值。
- 联合计算仅本池、零水位，范围 1–720 抽。六 / 五 / 四星产生 2000 / 200 / 20 配额，免费十连也产生配额且至少包含一位五星或以上。武器十连次数为 `floor(配额 / 1980)`。
- 干员模型不裁剪状态概率。联合模型逐步裁去总质量不超过 `1e-14` 的配额尾部，构建要求最低覆盖率不低于 `1 - 1e-9`，界面显示实际覆盖率。
- 显示图表的范围不影响完整分布的统计。反查使用最早达标的整数次数；达不到的目标明确显示未覆盖。

规则来源：[重构寻访](https://endfield.hypergryph.com/news/4776)、[加急招募保底说明](https://endfield.hypergryph.com/news/8564)、[特许寻访](https://endfield.hypergryph.com/news/6097)。

## 原有 Python / Streamlit

```powershell
python -m pip install -r requirements.txt
python precompute_joint_results.py
streamlit run streamlit_all_app.py
```

原有 Streamlit 演示地址：<https://endfield-draw-calculator.streamlit.app/>。

`myturnDRAW6up.py`、`character_markov.py` 和联合计算入口复用修正后的模型。联合缓存键包含模型版本，旧版缓存不会被当作新版结果；预计算会写入新的 `v2.0.0-*.parquet` 文件。Python API 在截断范围未覆盖完整结果时返回范围内完成的条件统计，未达到的分位点返回 `None`。

主要文件：`draw_model.py`（干员模型）、`joint_model.py`（联合模型）、`scripts/build_site.py`（构建）、`site/planner.mjs`（两池规划）、`site/app.mjs`（界面）、`.github/workflows/pages.yml`（部署）。
