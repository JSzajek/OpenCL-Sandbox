include "dependencies.lua"

project "VectorAddition"
	kind "ConsoleApp"

	language "C++"
	cppdialect "C++20"

	staticruntime "on"

	targetdir ("%{wks.location}/Binaries/" .. outputdir .. "/%{prj.name}")
	objdir ("%{wks.location}/Intermediates/" .. outputdir .. "/%{prj.name}")

	files
	{
		"src/**.h",
		"src/**.cpp",
		"shaders/*"
	}

	includedirs
	{
		"src",
		"../Utils/src",
	}
	
	libdirs
	{
	}
	
	links
	{
		"Utils"
	}

	LinkOpenCL()
	
	filter "system:windows"
		systemversion "latest"
	filter "configurations:Debug"
		symbols "On"
	filter "configurations:Release"
		optimize "On"
	filter "configurations:Dist"
		optimize "Full"