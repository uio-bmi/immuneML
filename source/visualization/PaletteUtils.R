generate_complex_heatmap_palette = function(annotations, palette) {
  for (column in colnames(annotations)) {
    column_values = annotations[, column, drop = TRUE]
    if (is.null(palette[[column]])) { # no palette specified
      if (!is.numeric(column_values) | all(column_values %in% c(0, 1))) { # discrete values or one-hot encoded
        palette[[column]] = generate_complex_heatmap_discrete_palette(column_values, palette[[column]])
      } else { # numeric values
        palette[[column]] = generate_complex_heatmap_continuous_palette(column_values, palette[[column]])
      }
    } else { # palette specified
      if ("colors" %in% names(palette[[column]])) { # continuous palette specified
        palette[[column]] = generate_complex_heatmap_continuous_palette(column_values, palette[[column]])
      } else { # discrete palette specified
        palette[[column]] = generate_complex_heatmap_discrete_palette(column_values, palette[[column]])
      }
    }
  }
  return(palette[colnames(annotations)])
}

generate_ggplot_palette = function(values, palette) {
  if (is.null(palette) || length(palette) == 0) { # no palette specified
    if (!is.numeric(values)) { # discrete values or one-hot encoded
      palette = generate_ggplot_discrete_palette(values, palette)
    } else { # numeric values
      palette = generate_ggplot_continuous_palette(values, palette)
    }
  } else { # palette specified
    if ("colors" %in% names(palette)) { # continuous palette specified
      palette = generate_ggplot_continuous_palette(values, palette)
    } else { # discrete palette specified
      palette = generate_ggplot_discrete_palette(values, palette)
    }
  }
}

generate_complex_heatmap_discrete_palette = function(values, palette) {
  palette = as.list(generate_discrete_palette(values, palette))
  if (all(values %in% c(0, 1))) {
    palette$`0` = "white"
    palette$`1` = "black"
  }
  return(unlist(palette))
}

generate_complex_heatmap_continuous_palette = function(values, palette) {
  palette = generate_continuous_palette(values, palette)
  palette = circlize::colorRamp2(breaks = palette$breaks, colors = palette$colors)
  return(palette)
}

generate_ggplot_discrete_palette = function(values, palette) {
  palette = generate_discrete_palette(values, palette)
  return(ggplot2::scale_color_manual(values = palette, aesthetics = c("colour", "fill")))
}

generate_ggplot_continuous_palette = function(values, palette) {
  palette = generate_continuous_palette(values, palette)
  palette = ggplot2::scale_color_gradientn(colors = palette$colors, values = scales::rescale(palette$breaks), limits = c(min(values, na.rm = TRUE), max(values, na.rm = TRUE)))
  return(palette)
}

generate_continuous_palette = function(values, palette) {
  if (is.null(palette)) palette = list() # in case no palette specified
  if ("colors" %in% names(palette)) { # colors specified
    if ("breaks" %in% names(palette)) { # both colors and breaks specified
      if (length(palette$colors) != length(palette$breaks)) { # colors and breaks do not match in length
        palette$breaks = seq(from = min(values), to = max(values), length.out = length(palette$colors))
      }
    } else { # only colors specified, breaks inferred
      palette$breaks = seq(from = min(values, na.rm = TRUE), to = max(values, na.rm = TRUE), length.out = length(palette$colors))
    }
  } else { # no palette specified
    palette$colors = c("white", "firebrick")
    palette$breaks = c(min(values, na.rm = TRUE), max(values, na.rm = TRUE))
  }
  return(palette)
}

generate_discrete_palette = function(values, palette) {
  if (is.null(palette)) palette = list()
  complete_palette = generate_default_discrete_palette(values)
  for (value in unique(values)) {
    if (value %in% names(palette)) {
      complete_palette[[as.character(value)]] = palette[[as.character(value)]]
    }
  }
  return(unlist(complete_palette))
}

generate_default_discrete_palette = function(values) {
  levels = unique(values[!values == "NA"])
  palette = c(ggsci::pal_aaas()(10), ggsci::pal_igv()(50))
  palette_custom = palette[1:length(levels)]
  names(palette_custom) = sort(levels)
  palette_custom = as.list(c(palette_custom, "NA" = "gray70"))
  return(palette_custom)
}
