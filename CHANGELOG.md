# Changelog
## [1.0.1] - 2023-08-21

### Refactored
- **Breaking:**: Simplified working with different variations of code through refactoring, such as specifying which version of the model to work with through adding new arguments, because our first generating script would misplace training data, which previously caused our model to generalize poorly. 

### Added
- Extended feature set to include additional features, working from an assumption that poor generalization was caused by lack of data
- Simplified overall pytorch model net architecture because of hardware limitations. 
- Added new arguments to main, see ReadMe for additional specification of appropriate usage.


### New MAE results
New MAE was much lower (21), however appeared to be more consistent with model's true performance on unseen data. However, the bad generalization still presist [in both centralized and federated approaches, more pronounced in federated approach]. When the tree-based learning was utilized, the accuracy was much better. The conclusion we have is that due to a stochastic nature of stochastic gradient descent, the model cannot generalize well. 
