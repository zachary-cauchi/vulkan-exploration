{ inputs, ... }:
{
  perSystem =
    { config
    , self'
    , pkgs
    , lib
    , ...
    }:
    {
      devShells.default = pkgs.mkShell rec {
        name = "vulkan-exploration-shell";
        inputsFrom = [
          self'.devShells.rust
          config.pre-commit.devShell # See ./nix/modules/pre-commit.nix
        ];

        buildInputs = with pkgs; [
          # Vulkan dependencies
          shader-slang
          shaderc
          spirv-tools
          vulkan-loader
          vulkan-tools
          vulkan-tools-lunarg
          vulkan-validation-layers

          libxcb

          # winit dependencies
          libx11
          libxcursor
          libxi
          libxkbcommon
          libxrandr
          wayland
        ];

        packages = with pkgs; [
          just
          nixd # Nix language server
          bacon
        ];

        LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
        VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
        # `shaderc_sys` does not detect NixOS paths properly. This is a workaround.
        # See https://github.com/google/shaderc-rs/issues/107#issuecomment-2592482347
        SHADERC_LIB_DIR = lib.makeLibraryPath [ pkgs.shaderc ];
      };
    };
}
