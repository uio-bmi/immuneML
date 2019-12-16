source(file.path(here::here(), "source", "visualization", "PaletteUtils.R"))
source(file.path(here::here(), "source", "visualization", "Heatmap.R"))

plot_density_heatmap = function(matrix,
                                feature_annotations = NULL,
                                palette = list(),
                                feature_names = NULL,
                                cluster_features = TRUE,
                                show_feature_dend = TRUE,
                                show_feature_names = TRUE,
                                show_legend_features = colnames(feature_annotations),
                                legend_position = "side",
                                text_size = 10,
                                feature_names_size = 4,
                                scale_features = FALSE,
                                height = 10,
                                width = 10,
                                result_name = "test_heatmap",
                                result_path = getwd()) {
  params = as.list(match.call())
  params[[1]] = NULL
  saveRDS(params, file.path(result_path, paste0(result_name, ".rds")))

  if (!is.null(feature_names))
    colnames(matrix) = as.character(feature_names)
  show_legend_features = as.character(show_legend_features)

  if (scale_features) {
    matrix = apply(
      matrix,
      MARGIN = 2,
      FUN = function(X)
        (X - min(X)) / diff(range(X))
    )
  }

  if (!is.list(palette))
    palette = rjson::fromJSON(palette)

  col_anno = create_annotation(feature_annotations,
                               "column",
                               show_legend_features,
                               text_size,
                               palette)

  heatmap = ComplexHeatmap::densityHeatmap(
    matrix,
    title = NULL,
    ylab = NULL,
    col = viridis::cividis(100),
    cluster_columns = cluster_features,
    show_column_dend = show_feature_dend,
    show_column_names = show_feature_names,
    column_names_side = "top",
    top_annotation = col_anno,
    column_names_gp = grid::gpar(fontsize = feature_names_size),
    heatmap_legend_param = list(
      title = "Value",
      title_gp = grid::gpar(fontsize = text_size, font_face = "plain"),
      labels_gp = grid::gpar(fontsize = 0.9 * text_size)
    )
  )

  pdf(
    file = file.path(result_path, paste0(result_name, ".pdf")),
    height = height,
    width = width
  )
  if (legend_position == "bottom") {
    ComplexHeatmap::draw(heatmap,
                         heatmap_legend_side = "bottom",
                         merge_legends = TRUE)
  } else {
    ComplexHeatmap::draw(heatmap)
  }

  dev.off()

  return(heatmap)
}
