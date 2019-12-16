source(file.path(here::here(), "source", "visualization", "PaletteUtils.R"))

plot_heatmap = function(matrix,
                        row_annotations = NULL,
                        column_annotations = NULL,
                        palette = list(),
                        row_names = NULL,
                        column_names = NULL,
                        cluster_rows = TRUE,
                        cluster_columns = TRUE,
                        show_row_dend = TRUE,
                        show_column_dend = TRUE,
                        show_row_names = TRUE,
                        show_column_names = TRUE,
                        show_legend_row = colnames(row_annotations),
                        show_legend_column = colnames(column_annotations),
                        legend_position = "side",
                        text_size = 10,
                        row_names_size = 4,
                        column_names_size = 4,
                        scale_rows = FALSE,
                        height = 10,
                        width = 10,
                        result_name = "test_heatmap",
                        result_path = getwd()) {

  params = as.list(match.call())
  params[[1]] = NULL
  saveRDS(params, file.path(result_path, paste0(result_name, ".rds")))

  if (!is.null(column_names)) colnames(matrix) = as.character(column_names)
  if (!is.null(row_names)) rownames(matrix) = as.character(row_names)
  show_legend_row = as.character(show_legend_row)
  show_legend_column = as.character(show_legend_column)

  if (scale_rows) {
    matrix = t(apply(t(matrix), MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X))))
  }

  if (!is.list(palette)) palette = rjson::fromJSON(palette)

  row_anno = create_annotation(row_annotations, "row", show_legend_row, text_size, palette)

  col_anno = create_annotation(column_annotations, "column", show_legend_column, text_size, palette)

  heatmap = ComplexHeatmap::Heatmap(
    matrix,
    col = viridis::cividis(100),
    cluster_rows = cluster_rows,
    cluster_columns = cluster_columns,
    show_row_dend = show_row_dend,
    show_column_dend = show_column_dend,
    show_row_names = show_row_names,
    show_column_names = show_column_names,
    row_names_side = "left",
    column_names_side = "top",
    top_annotation = col_anno,
    left_annotation = row_anno,
    column_names_gp = grid::gpar(fontsize = column_names_size),
    row_names_gp = grid::gpar(fontsize = row_names_size),
    heatmap_legend_param = list(title = "Value", title_gp = grid::gpar(fontsize = text_size, font_face = "plain"), labels_gp = grid::gpar(fontsize = 0.9 * text_size))
  )

  pdf(
    file = file.path(result_path, paste0(result_name, ".pdf")),
    height = height,
    width = width
  )
  if (legend_position == "bottom") {
    ComplexHeatmap::draw(heatmap, heatmap_legend_side = "bottom", merge_legends = TRUE)
  } else {
    ComplexHeatmap::draw(heatmap)
  }

  dev.off()

  return(heatmap)
}

create_annotation = function(data, which, show_legend, text_size, palette) {
  if (length(data) == 0) {
    result = NULL
  } else {
    result = ComplexHeatmap::HeatmapAnnotation(df = data,
                                               col = generate_complex_heatmap_palette(data, palette),
                                               show_annotation_name = TRUE,
                                               annotation_name_side = ifelse(which == "row", "bottom", "left"),
                                               which = which,
                                               show_legend = colnames(data) %in% show_legend,
                                               annotation_name_gp = grid::gpar(fontsize = text_size),
                                               annotation_legend_param = list(title_gp = grid::gpar(fontsize = text_size, font_face = "plain"), labels_gp = grid::gpar(fontsize = 0.9 * text_size)))
  }
  return(result)
}

test = function() {
  hi = matrix(1:15000, ncol = 500)
  row_annotation_data = data.frame(
    gene = c("v", "j", "v", "v", "j"),
    antigen = c("flu", "ebv", "ebv", "cmv", "ebv"),
    peptide = c("a", "a", "a", "b", "b"),
    something = c(0, 1, 1, 1, 0)
  )
  column_annotation_data = data.frame(
    disease = c("t1d", "control", "t1d"),
    age = c(1, 4, 2),
    sex = c("m", "f", "m")
  )
  column_annotation_data = column_annotation_data[rep(seq_len(nrow(column_annotation_data)), each=10),]
  row_annotation_data = row_annotation_data[rep(seq_len(nrow(row_annotation_data)), each=100), ]
  createRandString<- function() {
    v = c(sample(LETTERS, 5, replace = TRUE),
          sample(letters, 1, replace = TRUE))
    return(paste0(v,collapse = ""))
  }
  rownames(hi) = purrr::map_chr(1:nrow(hi), ~ createRandString())
  colnames(hi) = purrr::map_chr(1:ncol(hi), ~ createRandString())
  palette = list(disease = list(control = "white"), age = list(`1` = "royalblue"), sex = list(m = "black"))
  plot_heatmap(hi,
               column_annotation_data,
               row_annotation_data,
               show_row_names = TRUE,
               show_column_names = TRUE,
               column_names_size = 1,
               row_names_size = 7,
               text_size = 9,
               palette = palette,
               height = 6,
               width = 7,
               legend_position = "side")
}
